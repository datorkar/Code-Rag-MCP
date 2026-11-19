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

/// Windows: make sure the Windows ML runtime DLLs exist (bundled as resources)
/// and sit next to dev binaries in target/<profile> so `tauri dev` and
/// `cargo run -p rag-mcp` find them.
fn stage_winml() {
    if !std::env::var("TARGET").unwrap().contains("windows") {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap();
    let dir = manifest.join("binaries").join("winml");
    println!("cargo:rerun-if-changed={}", dir.join(".version").display());
    if !dir.join("onnxruntime.dll").is_file() {
        let ok = std::process::Command::new("node")
            .arg(root.join("scripts").join("fetch-winml.mjs"))
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            panic!("Windows ML DLLs missing; run `node scripts/fetch-winml.mjs`");
        }
    }
    let profile = std::env::var("PROFILE").unwrap();
    for out in [root.join("target").join(&profile), root.join("target").join(std::env::var("TARGET").unwrap()).join(&profile)] {
        if !out.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = f.path();
            if p.extension().is_some_and(|e| e == "dll") {
                let _ = std::fs::copy(&p, out.join(p.file_name().unwrap()));
            }
        }
    }
}

/// Windows preflight: perform the same copies tauri-build does (sidecar +
/// Windows ML DLL resources into the target dir) with explicit paths, so a
/// failure reports *which* path is missing instead of a bare OS error.
fn preflight_copies() {
    let triple = std::env::var("TARGET").unwrap();
    if !triple.contains("windows") {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Same derivation as tauri-build: OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out.ancestors().nth(3).expect("target dir from OUT_DIR").to_path_buf();
    println!("cargo:warning=preflight: target dir {}", target_dir.display());
    let mut copies = vec![(
        manifest.join("binaries").join(format!("rag-mcp-{triple}.exe")),
        target_dir.join("rag-mcp.exe"),
    )];
    for f in std::fs::read_dir(manifest.join("binaries").join("winml")).expect("binaries/winml").flatten() {
        let p = f.path();
        if p.extension().is_some_and(|e| e == "dll") {
            copies.push((p.clone(), target_dir.join(p.file_name().unwrap())));
        }
    }
    for (from, to) in copies {
        if let Err(e) = std::fs::copy(&from, &to) {
            panic!(
                "preflight copy failed: {} -> {}: {e} (from exists: {}, to dir exists: {})",
                from.display(),
                to.display(),
                from.exists(),
                to.parent().is_some_and(|d| d.is_dir())
            );
        }
    }
}

fn main() {
    stage_winml();
    preflight_copies();
    stage_sidecar();
    tauri_build::build()
}
