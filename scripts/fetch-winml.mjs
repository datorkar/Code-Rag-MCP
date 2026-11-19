// Downloads the pinned Windows ML NuGet package and extracts the x64 runtime
// DLLs (onnxruntime.dll, Microsoft.Windows.AI.MachineLearning.dll, DirectML.dll)
// into src-tauri/binaries/winml/. They ship next to the app executable.
// The package checksum is verified before anything is extracted.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync, copyFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

const PKG = "Microsoft.Windows.AI.MachineLearning";
const VERSION = "2.5.96"; // ships ONNX Runtime 1.28.3
const SHA256 = "35d6e84feb4142e8cecaf36704fa3ab0fc8251d892850fb80e8341f7d09653a3";
const ARCH = process.env.WINML_ARCH || "win-x64";
const DLLS = ["onnxruntime.dll", "Microsoft.Windows.AI.MachineLearning.dll", "DirectML.dll"];

const root = resolve(new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const outDir = join(root, "src-tauri", "binaries", "winml");
const stamp = join(outDir, ".version");

if (existsSync(stamp) && readFileSync(stamp, "utf8").trim() === `${VERSION}-${ARCH}` &&
    DLLS.every((d) => existsSync(join(outDir, d)))) {
  console.log(`winml: ${PKG} ${VERSION} already present`);
  process.exit(0);
}

const url = `https://www.nuget.org/api/v2/package/${PKG}/${VERSION}`;
console.log(`winml: downloading ${url}`);
const res = await fetch(url);
if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
const buf = Buffer.from(await res.arrayBuffer());
const sum = createHash("sha256").update(buf).digest("hex");
if (sum !== SHA256) throw new Error(`checksum mismatch: expected ${SHA256}, got ${sum}`);

const work = join(tmpdir(), `winml-${VERSION}-${process.pid}`);
rmSync(work, { recursive: true, force: true });
mkdirSync(work, { recursive: true });
const zip = join(work, "pkg.zip");
writeFileSync(zip, buf);
// Windows 10+ ships bsdtar (zip-capable); elsewhere use unzip.
if (process.platform === "win32") execFileSync("tar", ["-xf", zip, "-C", work], { stdio: "inherit" });
else execFileSync("unzip", ["-q", zip, "-d", work], { stdio: "inherit" });

const src = join(work, "runtimes", ARCH, "native");
mkdirSync(outDir, { recursive: true });
for (const f of readdirSync(outDir)) if (f.endsWith(".dll")) rmSync(join(outDir, f));
for (const d of DLLS) copyFileSync(join(src, d), join(outDir, d));
writeFileSync(stamp, `${VERSION}-${ARCH}\n`);
rmSync(work, { recursive: true, force: true });
console.log(`winml: extracted ${DLLS.join(", ")} -> ${outDir}`);
