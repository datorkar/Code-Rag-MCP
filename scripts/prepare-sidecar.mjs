// Builds `rag-mcp` in release mode and stages it for Tauri bundling:
//   src-tauri/binaries/rag-mcp-<triple>[.exe]   (externalBin -> next to the app exe)
//   src-tauri/binaries/ort/*                    (ONNX Runtime GPU provider libs -> next to the app exe)
//
// ONNX Runtime flavour follows the platform: Windows builds use Windows ML
// (DLLs fetched by scripts/fetch-winml.mjs), other platforms the bundled ORT
// with CUDA (CUDA 13 binaries; CPU fallback at runtime).
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, readdirSync, realpathSync, rmSync } from "node:fs";
import { join } from "node:path";

const triple =
  process.env.TAURI_ENV_TARGET_TRIPLE || execSync("rustc -vV").toString().match(/host: (\S+)/)[1];
const ext = triple.includes("windows") ? ".exe" : "";

if (triple.includes("windows")) execSync("node scripts/fetch-winml.mjs", { stdio: "inherit" });

execSync(["cargo", "build", "--release", "-p", "rag-mcp", "--target", triple].join(" "), {
  stdio: "inherit",
});

const outDir = join("target", triple, "release");
mkdirSync("src-tauri/binaries/ort", { recursive: true });
const dst = join("src-tauri", "binaries", `rag-mcp-${triple}${ext}`);
copyFileSync(join(outDir, `rag-mcp${ext}`), dst);
console.log(`sidecar: ${dst}`);

// Bundled-ORT builds (non-Windows): CUDA provider libs (symlinks into the ort
// cache; resolve them). TensorRT providers are skipped: huge and unused.
// Windows ML builds ship their DLLs from src-tauri/binaries/winml instead.
for (const f of readdirSync("src-tauri/binaries/ort")) {
  if (f !== "README.txt") rmSync(join("src-tauri/binaries/ort", f));
}
if (!triple.includes("windows")) {
  for (const f of readdirSync(outDir)) {
    if (/onnxruntime_providers_(cuda|shared)\.(dll|so)$/.test(f)) {
      try {
        copyFileSync(realpathSync(join(outDir, f)), join("src-tauri/binaries/ort", f));
        console.log(`ort lib: ${f}`);
      } catch (e) {
        console.warn(`ort lib ${f} skipped: ${e.message}`);
      }
    }
  }
}
