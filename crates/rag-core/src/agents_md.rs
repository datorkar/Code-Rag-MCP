//! Paste-ready AGENTS.md / CLAUDE.md paragraph telling a coding agent how to
//! use the `local-code-rag` MCP server for one specific project index.

use crate::config::IndexConfig;

pub const MCP_SERVER_NAME: &str = "local-code-rag";

/// `index` is the project's own index; `others` are listed as extra sources
/// (engine source, shared libraries, ...).
pub fn agents_md_snippet(index: &IndexConfig, others: &[IndexConfig]) -> String {
    let what = index.description.as_deref().unwrap_or(&index.name);
    let exts = index.allowed_extensions.join(" ");
    let id = &index.id;
    let mut s = format!(
        "## Code search (`{MCP_SERVER_NAME}` MCP)\n\n\
         This project ({what}) is indexed for semantic code search as `{id}`, rooted at `{root}` ({exts} files). \
         Before grepping or reading files broadly, call `search_index(index_id=\"{id}\", query=\"...\")` with exact \
         identifiers (class, function or variable names) and/or a short description of the behaviour you need; it \
         returns ranked `path:start-end` excerpts with paths relative to `{root}`, so open those files for full context. \
         Results are kept in sync automatically for small edits; after large changes (branch switches, big merges) \
         call `reindex_path(index_id=\"{id}\")`.",
        root = index.root_path,
    );
    let others: Vec<_> = others.iter().filter(|o| o.id != index.id).collect();
    if !others.is_empty() {
        let list = others
            .iter()
            .map(|o| format!("`{}` ({})", o.id, o.description.as_deref().unwrap_or(&o.name)))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!(
            " Other indexes you can search the same way: {list}; use them for engine or library code instead of \
             guessing APIs."
        ));
    }
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_names_index_root_and_others() {
        let game = IndexConfig {
            id: "my-game".into(),
            name: "My Unreal Project".into(),
            root_path: r"D:\Projects\MyUnrealGame".into(),
            allowed_extensions: vec![".cpp".into(), ".h".into()],
            ..Default::default()
        };
        let engine = IndexConfig {
            id: "ue5-runtime".into(),
            name: "Unreal Engine 5.8 Runtime".into(),
            ..Default::default()
        };
        let s = agents_md_snippet(&game, &[game.clone(), engine]);
        assert!(s.contains(r#"search_index(index_id="my-game""#));
        assert!(s.contains(r"`D:\Projects\MyUnrealGame`"));
        assert!(s.contains("`ue5-runtime` (Unreal Engine 5.8 Runtime)"));
        assert!(!s.contains("`my-game` (My Unreal Project)"), "own index not listed as 'other'");
        assert!(agents_md_snippet(&game, &[]).find("Other indexes").is_none());
    }
}
