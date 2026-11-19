use std::path::PathBuf;

/// Tauri's `externalBin` needs `binaries/rag-mcp-<triple>[.exe]` to exist at
/// compile time. Release bundles get it from `scripts/prepare-sidecar.mjs`
/// (run by `beforeBuildCommand`); for plain `cargo build` / `tauri dev` we
/// stage an already-built workspace `rag-mcp`, or a placeholder.
fn stage_sidecar() {
    let triple = std::env::var("TARGET").unwrap();
    let ext = if triple.contains("windows") { ".exe" } else { "" };
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let dst = manifest.join("binaries").join(format!("rag-mcp-{triple}{ext}"));
    let has_real = dst.metadata().map(|m| m.len() > 0).unwrap_or(false);
    if has_real {
        return;
    }
    let target = manifest.parent().unwrap().join("target");
    let profile = std::env::var("PROFILE").unwrap();
    let candidates = [
        target.join(&triple).join(&profile).join(format!("rag-mcp{ext}")),
        target.join(&profile).join(format!("rag-mcp{ext}")),
    ];
    std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
    if let Some(src) = candidates.iter().find(|p| p.is_file()) {
        std::fs::copy(src, &dst).unwrap();
    } else if !dst.exists() {
        println!(
            "cargo:warning=rag-mcp sidecar not built yet; staged an empty placeholder. \
             Run `cargo build -p rag-mcp` (or `node scripts/prepare-sidecar.mjs`) and rebuild."
        );
        std::fs::write(&dst, b"").unwrap();
    }
}

fn main() {
    stage_sidecar();
    tauri_build::build()
}
