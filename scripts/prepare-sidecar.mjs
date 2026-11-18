// Builds `rag-mcp` in release mode and copies it to src-tauri/binaries/ with
// the target-triple suffix Tauri's `externalBin` bundling expects. The
// bundler installs it next to the GUI executable as `rag-mcp(.exe)`.
import { execSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const features = process.env.RAG_FEATURES ? ["--features", process.env.RAG_FEATURES] : [];
const triple =
  process.env.TAURI_ENV_TARGET_TRIPLE ||
  execSync("rustc -vV").toString().match(/host: (\S+)/)[1];
const ext = triple.includes("windows") ? ".exe" : "";

execSync(["cargo", "build", "--release", "-p", "rag-mcp", "--target", triple, ...features].join(" "), {
  stdio: "inherit",
});
mkdirSync("src-tauri/binaries", { recursive: true });
const src = join("target", triple, "release", `rag-mcp${ext}`);
const dst = join("src-tauri", "binaries", `rag-mcp-${triple}${ext}`);
copyFileSync(src, dst);
console.log(`sidecar: ${src} -> ${dst}`);
