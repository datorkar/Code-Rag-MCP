//! Trim a retrieved chunk to the window of lines that best matches the query,
//! so Claude gets the relevant part without the whole 3000-character chunk.

use crate::store::SearchHit;

/// Lower-cased query terms (identifiers split on `::`, `.`, spaces, ...),
/// plus the camelCase / PascalCase parts of each identifier.
fn terms(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in query.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if word.len() >= 3 {
            out.push(word.to_lowercase());
        }
        let mut cur = String::new();
        let chars: Vec<char> = word.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            let boundary = i > 0
                && c.is_uppercase()
                && (chars[i - 1].is_lowercase()
                    || chars.get(i + 1).is_some_and(|n| n.is_lowercase()));
            if boundary || c == '_' {
                if cur.len() >= 4 {
                    out.push(cur.to_lowercase());
                }
                cur.clear();
            }
            if c != '_' {
                cur.push(c);
            }
        }
        if cur.len() >= 4 && cur.len() < word.len() {
            out.push(cur.to_lowercase());
        }
    }
    out.sort();
    out.dedup();
    out
}

pub fn trim_hit(mut hit: SearchHit, query: &str, max_lines: usize) -> SearchHit {
    let lines: Vec<&str> = hit.text.lines().collect();
    if max_lines == 0 || lines.len() <= max_lines {
        return hit;
    }
    let terms = terms(query);
    let scores: Vec<usize> = lines
        .iter()
        .map(|l| {
            let l = l.to_lowercase();
            terms.iter().filter(|t| l.contains(t.as_str())).count()
        })
        .collect();

    // Sliding window with the highest total score (earliest on ties; with no
    // matches this keeps the chunk head, usually a declaration).
    let mut best = (0usize, 0usize);
    let mut sum: usize = scores[..max_lines].iter().sum();
    best.1 = sum;
    for start in 1..=lines.len() - max_lines {
        sum = sum + scores[start + max_lines - 1] - scores[start - 1];
        if sum > best.1 {
            best = (start, sum);
        }
    }
    // Centre the window on the matching lines inside it.
    let mut start = best.0;
    let win = &scores[start..start + max_lines];
    if let (Some(first), Some(last)) = (win.iter().position(|&x| x > 0), win.iter().rposition(|&x| x > 0)) {
        let centre = start + (first + last) / 2;
        start = centre.saturating_sub(max_lines / 2).min(lines.len() - max_lines);
    }
    let chunk = (hit.start_line, hit.end_line);
    hit.text = lines[start..start + max_lines].join("\n");
    hit.start_line = chunk.0 + start as u32;
    hit.end_line = hit.start_line + max_lines as u32 - 1;
    hit.chunk_lines = Some(chunk);
    hit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_identifiers() {
        let t = terms("UCharacterMovementComponent::PhysWalking jump");
        for w in ["ucharactermovementcomponent", "physwalking", "character", "movement", "component", "walking", "jump"] {
            assert!(t.contains(&w.to_string()), "{w} missing from {t:?}");
        }
    }

    #[test]
    fn picks_matching_window() {
        let mut text = String::new();
        for i in 0..100 {
            if i == 70 {
                text.push_str("void UFoo::PhysWalking(float DeltaTime)\n");
            } else {
                text.push_str(&format!("int Filler{i};\n"));
            }
        }
        let hit = SearchHit {
            path: "a.cpp".into(),
            start_line: 11,
            end_line: 110,
            score: 1.0,
            text,
            context: String::new(),
            chunk_lines: None,
        };
        let t = trim_hit(hit, "PhysWalking", 20);
        assert_eq!(t.text.lines().count(), 20);
        assert!(t.text.contains("PhysWalking"));
        assert_eq!(t.chunk_lines, Some((11, 110)));
        assert_eq!(t.start_line, 11 + 70 - 10); // match centred in the window
        assert_eq!(t.end_line, t.start_line + 19);
    }
}
