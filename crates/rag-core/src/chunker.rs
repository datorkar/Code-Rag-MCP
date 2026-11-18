//! AST-aware chunking (tree-sitter via `text-splitter`) with a plain-text fallback.

use text_splitter::{ChunkConfig, CodeSplitter, TextSplitter};
use tree_sitter::Language;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub rel_path: String,
    /// 1-based, inclusive.
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

impl Chunk {
    /// Text fed to the embedding model: the path gives the model useful
    /// context (module / class names) that is often absent from the body.
    pub fn embedding_input(&self) -> String {
        format!("// File: {}\n{}", self.rel_path, self.text)
    }
}

fn language_for(rel_path: &str) -> Option<Language> {
    let ext = rel_path.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        // HLSL (.usf/.ush) is C-like enough for the C++ grammar to find good boundaries.
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "inl" | "ipp" | "usf"
        | "ush" | "hlsl" => tree_sitter_cpp::LANGUAGE.into(),
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "cs" => tree_sitter_c_sharp::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        _ => return None,
    })
}

/// Byte offsets where each line starts.
fn line_starts(text: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(text.match_indices('\n').map(|(i, _)| i + 1))
        .collect()
}

fn line_of(starts: &[usize], byte: usize) -> u32 {
    (starts.partition_point(|&s| s <= byte)) as u32
}

pub fn chunk_text(rel_path: &str, text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    let config = || {
        ChunkConfig::new(chunk_size)
            .with_overlap(overlap.min(chunk_size.saturating_sub(1)))
            .expect("overlap < size")
            .with_trim(true)
    };
    let pieces: Vec<(usize, &str)> = match language_for(rel_path)
        .and_then(|lang| CodeSplitter::new(lang, config()).ok())
    {
        Some(splitter) => splitter.chunk_indices(text).collect(),
        None => TextSplitter::new(config()).chunk_indices(text).collect(),
    };

    let starts = line_starts(text);
    pieces
        .into_iter()
        .filter(|(_, s)| !s.trim().is_empty())
        .map(|(off, s)| {
            let end_byte = off + s.len().saturating_sub(1);
            Chunk {
                rel_path: rel_path.to_string(),
                start_line: line_of(&starts, off),
                end_line: line_of(&starts, end_byte),
                text: s.to_string(),
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
}
