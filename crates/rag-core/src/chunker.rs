//! AST-aware chunking (tree-sitter via `text-splitter`) with a plain-text fallback.

use std::collections::HashSet;

use text_splitter::{ChunkConfig, CodeSplitter, TextSplitter};
use tree_sitter::{Language, Node, Parser};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub rel_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    /// Enclosing scopes at the chunk start, e.g.
    /// `namespace UE > class UFoo : public UObject > void UFoo::Tick(float DeltaTime)`.
    pub context: String,
}

impl Chunk {
    /// What the embedding model sees: path + enclosing scopes + code.
    pub fn embedding_input(&self) -> String {
        if self.context.is_empty() {
            format!("// File: {}\n{}", self.rel_path, self.text)
        } else {
            format!("// File: {}\n// In: {}\n{}", self.rel_path, self.context, self.text)
        }
    }

    /// What the keyword index sees: code + scopes + split identifiers
    /// (`PhysWalking` -> `phys walking`).
    pub fn fts_input(&self) -> String {
        let split = crate::idents::split_terms(&format!("{}\n{}", self.context, self.text));
        format!("{}\n{}\n{}", self.text, self.context, split)
    }
}

fn language_for(rel_path: &str) -> Option<Language> {
    let ext = rel_path.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "inl" | "ipp" | "usf"
        | "ush" | "hlsl" => tree_sitter_cpp::LANGUAGE.into(),
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "cs" => tree_sitter_c_sharp::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        _ => return None,
    })
}

/// Node kinds that name a scope (all grammars; kinds are distinct enough).
const SCOPE_KINDS: &[&str] = &[
    // C / C++
    "namespace_definition", "class_specifier", "struct_specifier", "union_specifier", "enum_specifier",
    "function_definition",
    // Rust
    "mod_item", "impl_item", "trait_item", "struct_item", "enum_item", "function_item",
    // C#
    "namespace_declaration", "file_scoped_namespace_declaration", "class_declaration", "struct_declaration",
    "interface_declaration", "record_declaration", "enum_declaration", "method_declaration",
    "constructor_declaration", "property_declaration",
    // Python
    "class_definition",
];

const MAX_LABEL: usize = 120;
const MAX_CONTEXT: usize = 200;

/// Signature of a scope node: its text up to the body, whitespace collapsed.
fn label(node: Node, src: &str) -> Option<String> {
    let end = node
        .child_by_field_name("body")
        .map(|b| b.start_byte())
        .unwrap_or_else(|| node.end_byte().min(node.start_byte() + 2 * MAX_LABEL));
    let raw = src.get(node.start_byte()..end)?;
    let raw = raw.split('{').next().unwrap_or(raw);
    let mut out = String::new();
    for w in raw.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    let out = out.trim_end_matches([':', '{', ';']).trim().to_string();
    if out.is_empty() {
        return None;
    }
    Some(if out.chars().count() > MAX_LABEL {
        let mut t: String = out.chars().take(MAX_LABEL - 1).collect();
        t.push('…');
        t
    } else {
        out
    })
}

/// Enclosing scopes of byte `at`, outermost first, trimmed to `MAX_CONTEXT`
/// (innermost scopes are kept).
fn context_at(root: Node, src: &str, at: usize) -> String {
    let Some(mut node) = root.descendant_for_byte_range(at, at) else { return String::new() };
    let mut labels = Vec::new();
    loop {
        if SCOPE_KINDS.contains(&node.kind()) {
            if let Some(l) = label(node, src) {
                labels.push(l);
            }
        }
        match node.parent() {
            Some(p) => node = p,
            None => break,
        }
    }
    // labels: innermost first. Keep as many inner ones as fit.
    let mut kept: Vec<String> = Vec::new();
    let mut len = 0;
    for l in labels {
        if len + l.len() + 3 > MAX_CONTEXT && !kept.is_empty() {
            break;
        }
        len += l.len() + 3;
        kept.push(l);
    }
    kept.reverse();
    kept.join(" > ")
}

fn line_starts(text: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(text.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

fn line_of(starts: &[usize], byte: usize) -> u32 {
    (starts.partition_point(|&s| s <= byte)) as u32
}

pub fn chunk_text(rel_path: &str, text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    chunk_file(rel_path, text, chunk_size, overlap, &HashSet::new())
}

/// Chunk one file. A leading header whose hash is in `boilerplate` (licence
/// text shared by many files) is skipped; line numbers stay file-relative.
pub fn chunk_file(
    rel_path: &str,
    text: &str,
    chunk_size: usize,
    overlap: usize,
    boilerplate: &HashSet<String>,
) -> Vec<Chunk> {
    let config = || {
        ChunkConfig::new(chunk_size)
            .with_overlap(overlap.min(chunk_size.saturating_sub(1)))
            .expect("overlap < size")
            .with_trim(true)
    };
    let skip = crate::boilerplate::content_start(rel_path, text, boilerplate);
    let body = &text[skip..];
    let lang = language_for(rel_path);
    let pieces: Vec<(usize, &str)> = match lang.clone().and_then(|lang| CodeSplitter::new(lang, config()).ok()) {
        Some(splitter) => splitter.chunk_indices(body).collect(),
        None => TextSplitter::new(config()).chunk_indices(body).collect(),
    };
    // Parse once more for scope context (the splitter's tree is not exposed).
    let tree = lang.and_then(|l| {
        let mut p = Parser::new();
        p.set_language(&l).ok()?;
        p.parse(body, None)
    });

    let starts = line_starts(text);
    pieces
        .into_iter()
        .filter(|(_, s)| !s.trim().is_empty())
        .map(|(off, s)| {
            let end_byte = off + s.len().saturating_sub(1);
            let context = tree.as_ref().map(|t| context_at(t.root_node(), body, off)).unwrap_or_default();
            Chunk {
                rel_path: rel_path.to_string(),
                start_line: line_of(&starts, skip + off),
                end_line: line_of(&starts, skip + end_byte),
                text: s.to_string(),
                context,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpp_chunks_have_correct_lines() {
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!(
                "void Func{i}(int A)\n{{\n    DoSomething(A, {i});\n    DoOther();\n}}\n\n"
            ));
        }
        let chunks = chunk_text("Source/Foo.cpp", &src, 200, 20);
        assert!(chunks.len() > 5);
        let lines: Vec<&str> = src.lines().collect();
        for c in &chunks {
            assert!(c.text.len() <= 200);
            assert!(c.start_line >= 1 && c.end_line >= c.start_line);
            let first = c.text.lines().next().unwrap();
            assert!(
                lines[c.start_line as usize - 1].contains(first.trim()),
                "line mismatch for {:?}",
                c
            );
        }
        assert!(chunks[0].text.starts_with("void Func0"));
    }

    #[test]
    fn unknown_extension_falls_back_to_text() {
        let src = "[Section]\nKey=Value\n".repeat(50);
        let chunks = chunk_text("Config/DefaultEngine.ini", &src, 150, 0);
        assert!(chunks.len() > 3);
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn cpp_context_names_enclosing_scopes() {
        let src = "namespace Game {\nclass AFoo : public AActor {\npublic:\n    void Tick(float DeltaTime) {\n        int a = 0;\n        a += 1;\n        a += 2;\n        a += 3;\n    }\n};\n}\n\nvoid AFoo::Jump(int Height)\n{\n    DoJump(Height);\n    DoJump(Height);\n}\n";
        let chunks = chunk_text("Foo.cpp", src, 60, 0);
        let tick = chunks.iter().find(|c| c.text.contains("a += 2")).unwrap();
        assert!(tick.context.contains("namespace Game"), "{:?}", tick.context);
        assert!(tick.context.contains("class AFoo"), "{:?}", tick.context);
        assert!(tick.context.contains("void Tick(float DeltaTime)"), "{:?}", tick.context);
        let jump = chunks.iter().find(|c| c.text.contains("DoJump")).unwrap();
        assert!(jump.context.contains("AFoo::Jump(int Height)"), "{:?}", jump.context);
        assert!(jump.embedding_input().contains("// In: "));
        assert!(jump.fts_input().contains("do jump"));
    }

    #[test]
    fn boilerplate_header_is_skipped_with_correct_lines() {
        let src = "// Copyright Epic Games, Inc. All Rights Reserved.\n\n#pragma once\n\nint Answer() { return 42; }\n";
        let (_, h) = crate::boilerplate::header_block("A.h", src).unwrap();
        let set: HashSet<String> = [h].into_iter().collect();
        let chunks = chunk_file("A.h", src, 500, 0, &set);
        assert!(!chunks[0].text.contains("Copyright"));
        assert_eq!(chunks[0].start_line, 3);
        assert!(chunks[0].text.starts_with("#pragma once"));
    }
}
