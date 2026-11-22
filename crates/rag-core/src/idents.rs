//! Identifier splitting for keyword search: `UCharacterMovementComponent`
//! -> `character movement component`, `phys_walking` -> `phys walking`, so
//! BM25 matches natural-language words inside compound identifiers.

/// Parts of one identifier (camelCase / PascalCase / snake_case / digits),
/// lower-cased. A single-part identifier yields nothing.
pub fn split_identifier(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut parts = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if c == '_' || c == '$' {
            if !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let boundary = i > 0
            && !cur.is_empty()
            && ((c.is_uppercase()
                && (chars[i - 1].is_lowercase()
                    || chars[i - 1].is_ascii_digit()
                    || (chars[i - 1].is_uppercase() && chars.get(i + 1).is_some_and(|n| n.is_lowercase()))))
                || (c.is_ascii_digit() != chars[i - 1].is_ascii_digit() && chars[i - 1] != '_'));
        if boundary {
            parts.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    if parts.len() < 2 {
        return Vec::new();
    }
    parts
        .into_iter()
        .filter(|p| p.len() >= 2 && !p.chars().all(|c| c.is_ascii_digit()))
        .map(|p| p.to_lowercase())
        .collect()
}

/// Space-separated unique parts of every compound identifier in `text`.
pub fn split_terms(text: &str) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out = String::new();
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if word.len() < 4 {
            continue;
        }
        for p in split_identifier(word) {
            if seen.insert(p.clone()) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(&p);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_common_styles() {
        assert_eq!(split_identifier("UCharacterMovementComponent"), ["character", "movement", "component"]);
        assert_eq!(split_identifier("PhysWalking"), ["phys", "walking"]);
        assert_eq!(split_identifier("phys_walking"), ["phys", "walking"]);
        assert_eq!(split_identifier("FVector3d"), ["vector"]);
        assert_eq!(split_identifier("HTTPServer"), ["http", "server"]);
        assert!(split_identifier("Tick").is_empty());
        assert!(split_terms("void UCharacterMovementComponent::PhysWalking()").contains("movement"));
    }
}
