// Builds `rag-mcp` in release mode and stages it for Tauri bundling:
//   src-tauri/binaries/rag-mcp-<triple>[.exe]   (externalBin -> next to the app exe)
//   src-tauri/binaries/ort/*                    (ONNX Runtime GPU provider libs -> next to the app exe)
//
// Features: CUDA by default. Override with RAG_FEATURES, e.g.
//   RAG_FEATURES=none      CPU only
//   RAG_FEATURES=directml  DirectML instead of CUDA
// The CUDA 12 vs 13 build of ONNX Runtime is auto-detected from CUDA_HOME;
// force it with ORT_CUDA_VERSION=12|13.
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync, readdirSync, realpathSync, rmSync } from "node:fs";
import { join } from "node:path";

const feat = process.env.RAG_FEATURES;
const featureArgs =
  feat === "none" ? ["--no-default-features"] : feat ? ["--no-default-features", "--features", feat] : [];
const host = execSync("rustc -vV").toString().match(/host: (\S+)/)[1];
const triple = process.env.TAURI_ENV_TARGET_TRIPLE || host;
// Build into the same target dir as `tauri build` (no --target for the host)
// so rag-mcp and the app share one compiled dependency graph.
const targetArgs = triple === host ? [] : ["--target", triple];
const ext = triple.includes("windows") ? ".exe" : "";

execSync(["cargo", "build", "--release", "-p", "rag-mcp", ...targetArgs, ...featureArgs].join(" "), {
  stdio: "inherit",
});

const outDir = triple === host ? join("target", "release") : join("target", triple, "release");
mkdirSync("src-tauri/binaries/ort", { recursive: true });
const dst = join("src-tauri", "binaries", `rag-mcp-${triple}${ext}`);
copyFileSync(join(outDir, `rag-mcp${ext}`), dst);
console.log(`sidecar: ${dst}`);

// Provider libs (symlinks into the ort cache on Unix; resolve them). TensorRT
// providers are skipped: they are huge and unused.
for (const f of readdirSync("src-tauri/binaries/ort")) {
  if (f !== "README.txt") rmSync(join("src-tauri/binaries/ort", f));
}
for (const f of readdirSync(outDir)) {
  if (/onnxruntime_providers_(cuda|shared)\.(dll|so)$/.test(f) || /^DirectML\.dll$/i.test(f)) {
    copyFileSync(realpathSync(join(outDir, f)), join("src-tauri/binaries/ort", f));
    console.log(`ort lib: ${f}`);
  }
}
