//! Detects licence / copyright headers that repeat across many files of an
//! index (e.g. `// Copyright Epic Games, Inc. All Rights Reserved.`), so they
//! are not embedded or keyword-indexed. Generic: no patterns are hard-coded;
//! a leading comment block counts as boilerplate when the same (normalised)
//! block starts many files.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;

/// Bytes read per file when collecting header statistics.
const HEAD_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy)]
enum Syntax {
    /// `//`, `/* */` (C, C++, C#, Rust, JS, shaders, ...). `#` lines are code.
    CLike,
    /// `#` comments (Python, shell, ini, toml, yaml, cmake, PowerShell).
    Hash,
    /// `--` comments (Lua, SQL).
    Dash,
    /// `<!-- -->` (XML, HTML, Markdown).
    Markup,
}

fn syntax_for(rel_path: &str) -> Option<Syntax> {
    let ext = rel_path.rsplit('.').next()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "c" | "cc" | "cpp" | "cxx" | "h" | "hh" | "hpp" | "hxx" | "inl" | "ipp" | "usf" | "ush" | "hlsl"
        | "glsl" | "vert" | "frag" | "comp" | "rs" | "cs" | "js" | "jsx" | "ts" | "tsx" | "mjs" | "java"
        | "kt" | "go" | "swift" | "m" | "mm" | "dart" | "scala" | "php" | "css" | "scss" => Syntax::CLike,
        "py" | "sh" | "bash" | "ps1" | "psm1" | "ini" | "toml" | "yaml" | "yml" | "cmake" | "rb" | "pl" => {
            Syntax::Hash
        }
        "lua" | "sql" => Syntax::Dash,
        "xml" | "html" | "htm" | "md" | "xaml" | "vcxproj" | "props" => Syntax::Markup,
        _ => return None,
    })
}

/// The file's leading comment block: `(byte offset where code starts,
/// normalised hash)`. `None` when the file does not start with a comment.
pub fn header_block(rel_path: &str, text: &str) -> Option<(usize, String)> {
    let syntax = syntax_for(rel_path)?;
    let mut norm = String::new();
    let mut in_block = false;
    let mut offset = 0usize;
    let mut end = 0usize;
    let mut comment_lines = 0usize;
    // Skip a UTF-8 BOM.
    let body = text.strip_prefix('\u{feff}').unwrap_or(text);
    offset += text.len() - body.len();
    for line in body.split_inclusive('\n') {
        let t = line.trim();
        let is_comment = if in_block {
            let close = match syntax {
                Syntax::Markup => "-->",
                _ => "*/",
            };
            if t.contains(close) {
                in_block = false;
            }
            true
        } else if t.is_empty() {
            false
        } else {
            match syntax {
                Syntax::CLike => {
                    if t.starts_with("//") {
                        true
                    } else if t.starts_with("/*") {
                        in_block = !t[2..].contains("*/");
                        true
                    } else {
                        break;
                    }
                }
                Syntax::Hash => {
                    if t.starts_with('#') {
                        true
                    } else {
                        break;
                    }
                }
                Syntax::Dash => {
                    if t.starts_with("--") {
                        true
                    } else {
                        break;
                    }
                }
                Syntax::Markup => {
                    if t.starts_with("<!--") {
                        in_block = !t.contains("-->");
                        true
                    } else {
                        break;
                    }
                }
            }
        };
        offset += line.len();
        if is_comment {
            comment_lines += 1;
            end = offset;
            norm.push_str(&normalise(t));
            norm.push('\n');
        } else if t.is_empty() {
            // Blank lines inside / after the header belong to it.
            if comment_lines > 0 {
                end = offset;
            }
        }
    }
    if comment_lines == 0 || norm.trim().is_empty() {
        return None;
    }
    Some((end, blake3::hash(norm.as_bytes()).to_hex()[..16].to_string()))
}

/// Years and year ranges -> `#` (they differ between files), whitespace collapsed.
fn normalise(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut space = false;
    for c in line.chars() {
        if c.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        if c.is_ascii_digit() {
            // A digit run (a year) becomes one `#`; `#-#` (a range) too.
            if !out.ends_with('#') {
                if out.ends_with("#-") {
                    out.pop();
                } else {
                    out.push('#');
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// A header is boilerplate when it starts at least this many of `n` files.
pub fn threshold(n_files: usize) -> usize {
    (n_files / 20).max(20)
}

/// Read the head of every file and return the hashes of headers shared by
/// enough files to be boilerplate. Runs on the current rayon pool.
pub fn detect<'a>(files: impl rayon::iter::IntoParallelIterator<Item = (&'a str, &'a Path)>) -> Vec<String> {
    use rayon::prelude::*;
    // (files considered, header hash -> count)
    let (n, counts): (usize, HashMap<String, usize>) = files
        .into_par_iter()
        .filter_map(|(rel, path)| {
            syntax_for(rel)?;
            let mut buf = Vec::with_capacity(HEAD_BYTES);
            std::fs::File::open(path).ok()?.take(HEAD_BYTES as u64).read_to_end(&mut buf).ok()?;
            let text = String::from_utf8_lossy(&buf);
            Some(header_block(rel, &text).map(|(_, h)| h))
        })
        .fold(
            || (0usize, HashMap::new()),
            |(n, mut m), h| {
                if let Some(h) = h {
                    *m.entry(h).or_insert(0) += 1;
                }
                (n + 1, m)
            },
        )
        .reduce(
            || (0, HashMap::new()),
            |(na, mut a), (nb, b)| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                (na + nb, a)
            },
        );
    let min = threshold(n);
    let mut out: Vec<String> = counts.into_iter().filter(|(_, c)| *c >= min).map(|(h, _)| h).collect();
    out.sort();
    out
}

/// Offset where content starts after a boilerplate header (0 if none).
pub fn content_start(rel_path: &str, text: &str, boilerplate: &HashSet<String>) -> usize {
    if boilerplate.is_empty() {
        return 0;
    }
    match header_block(rel_path, text) {
        Some((end, h)) if boilerplate.contains(&h) => end,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_header_with_different_years_matches() {
        let a = "// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.\n\n#pragma once\nclass A {};\n";
        let b = "// Copyright  2024 Epic Games, Inc. All Rights Reserved.\n#include \"B.h\"\n";
        let (ea, ha) = header_block("A.h", a).unwrap();
        let (_, hb) = header_block("B.cpp", b).unwrap();
        assert_eq!(ha, hb);
        assert!(a[ea..].starts_with("#pragma once"));
    }

    #[test]
    fn block_comments_and_code_start() {
        let s = "/*\n * Licensed under MIT\n */\nint x;\n";
        let (end, _) = header_block("x.c", s).unwrap();
        assert_eq!(&s[end..], "int x;\n");
        assert!(header_block("x.c", "int x; // not a header\n").is_none());
        let py = "#!/usr/bin/env python\n# Copyright Foo\nimport os\n";
        let (end, _) = header_block("x.py", py).unwrap();
        assert_eq!(&py[end..], "import os\n");
    }

    #[test]
    fn content_start_only_for_known_boilerplate() {
        let s = "// Copyright X\nint a;\n";
        let (_, h) = header_block("a.cpp", s).unwrap();
        let set: HashSet<String> = [h].into_iter().collect();
        assert_eq!(&s[content_start("a.cpp", s, &set)..], "int a;\n");
        assert_eq!(content_start("a.cpp", "// Unique doc\nint a;\n", &set), 0);
    }
}
