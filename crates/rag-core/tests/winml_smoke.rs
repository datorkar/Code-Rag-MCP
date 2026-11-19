//! Windows ML smoke test: loads onnxruntime.dll from the Windows ML package,
//! lists devices and catalog providers, and embeds text on CPU (must work) and
//! via the GPU path (DirectML / vendor EP; reported, not required — CI runners
//! have no GPU). Run with:
//!   node scripts/fetch-winml.mjs
//!   set LOCAL_CODE_RAG_WINML_DIR=<repo>\src-tauri\binaries\winml
//!   cargo test -p rag-core --features winml --test winml_smoke -- --ignored --nocapture
#![cfg(all(feature = "winml", windows))]

use rag_core::embedder::Embedder;
use rag_core::{runtime, ExecutionProvider};

#[test]
#[ignore]
fn winml_runtime_embeds() {
    runtime::init().expect("Windows ML ONNX Runtime loads");
    let info = runtime::info().unwrap();
    println!("runtime: {info:#?}");
    assert_eq!(info.flavour, "winml");
    assert!(info.devices.iter().any(|d| d.kind == "cpu"), "CPU device listed");

    match runtime::winml_providers() {
        Ok(list) => println!("Windows ML catalog: {list:#?}"),
        Err(e) => println!("Windows ML catalog unavailable here: {e:#}"),
    }

    let cache = tempfile::tempdir().unwrap();
    let texts = vec!["void UHealthComponent::ApplyDamage(float Amount)".to_string(), "fn main() {}".to_string()];

    let mut cpu = Embedder::load("BAAI/bge-small-en-v1.5", ExecutionProvider::Cpu, cache.path(), 256)
        .expect("CPU embedder loads");
    let v = cpu.embed_documents(&texts, 2).expect("CPU embedding works");
    assert_eq!(v.len(), 2);
    assert_eq!(v[0].len(), 384);

    match Embedder::load("BAAI/bge-small-en-v1.5", ExecutionProvider::Gpu, cache.path(), 256)
        .and_then(|mut e| e.embed_documents(&texts, 2))
    {
        Ok(g) => {
            let cos: f32 = v[0].iter().zip(&g[0]).map(|(a, b)| a * b).sum::<f32>()
                / (v[0].iter().map(|a| a * a).sum::<f32>().sqrt() * g[0].iter().map(|a| a * a).sum::<f32>().sqrt());
            println!("GPU path OK, cosine(cpu, gpu) = {cos:.4}");
            assert!(cos > 0.99, "GPU and CPU embeddings agree");
        }
        Err(e) => println!("GPU path unavailable on this machine: {e:#}"),
    }
}
