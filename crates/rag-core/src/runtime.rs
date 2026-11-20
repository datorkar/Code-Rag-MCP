//! ONNX Runtime bootstrap shared by the GUI and the MCP server.
//!
//! Exactly one ORT environment may exist per process; it is created here and
//! handed to every FastEmbed model.
//!
//! * `bundled` builds: ORT is statically linked; the optional `cuda` feature adds
//!   the CUDA execution provider (CPU fallback when CUDA/cuDNN are missing).
//! * `winml` builds (Windows): ORT is loaded from the Windows ML package
//!   (`onnxruntime.dll`, `Microsoft.Windows.AI.MachineLearning.dll`, `DirectML.dll`
//!   next to the executable). Vendor execution providers that Windows ML has
//!   already installed are registered at startup; GPU sessions use ORT's
//!   "prefer GPU" device policy when a vendor EP is present, DirectML otherwise.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use fastembed::ExecutionProviderDispatch;
use ort::environment::Environment;
use ort::session::builder::AutoDevicePolicy;
use serde::Serialize;

use crate::config::ExecutionProvider;

#[cfg(not(any(feature = "bundled", feature = "winml")))]
compile_error!("rag-core needs an ONNX Runtime flavour: enable feature `bundled` (or `cuda`) or `winml`");

#[cfg(all(feature = "bundled", feature = "winml"))]
compile_error!("features `bundled` and `winml` are mutually exclusive");

pub const FLAVOUR: &str = if cfg!(feature = "winml") { "winml" } else { "bundled" };

/// Env var overriding where the Windows ML DLLs are looked up.
pub const WINML_DIR_ENV: &str = "LOCAL_CODE_RAG_WINML_DIR";

struct State {
    env: Environment,
    /// Plugin EPs registered with `env` (by name), to avoid double registration.
    registered: Mutex<Vec<String>>,
    #[allow(dead_code)]
    dll_dir: Option<PathBuf>,
}

static STATE: OnceLock<std::result::Result<State, String>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    /// Execution provider name, e.g. `CPUExecutionProvider`, `DmlExecutionProvider`, `NvTensorRTRTXExecutionProvider`.
    pub ep: String,
    pub ep_vendor: String,
    /// `cpu` | `gpu` | `npu`
    pub kind: String,
    /// Hardware vendor (e.g. NVIDIA, AMD, Intel).
    pub vendor: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeInfo {
    pub flavour: &'static str,
    pub dll_dir: Option<String>,
    pub registered_providers: Vec<String>,
    pub devices: Vec<DeviceInfo>,
    /// Vendor (non-DirectML, non-CPU) GPU execution provider available.
    pub vendor_gpu: bool,
}

/// Directory holding the Windows ML DLLs: `$LOCAL_CODE_RAG_WINML_DIR`, the
/// directory of `$ORT_DYLIB_PATH`, or the executable's directory.
#[cfg(feature = "winml")]
pub fn winml_dir() -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = [
        std::env::var_os(WINML_DIR_ENV).map(PathBuf::from),
        std::env::var_os("ORT_DYLIB_PATH").and_then(|p| PathBuf::from(p).parent().map(|d| d.to_path_buf())),
        std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())),
    ]
    .into_iter()
    .flatten()
    .collect();
    let dll = if cfg!(windows) { "onnxruntime.dll" } else { "libonnxruntime.so" };
    candidates
        .iter()
        .find(|d| d.join(dll).is_file())
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "{dll} (Windows ML) not found in {}; run `node scripts/fetch-winml.mjs` or set {WINML_DIR_ENV}",
                candidates.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
            )
        })
}

fn build_state() -> Result<State> {
    #[cfg(feature = "winml")]
    {
        let dir = winml_dir()?;
        let dll = dir.join(if cfg!(windows) { "onnxruntime.dll" } else { "libonnxruntime.so" });
        tracing::info!("loading Windows ML ONNX Runtime from {}", dll.display());
        let env = ort::init_from(&dll)
            .map_err(|e| anyhow!("loading {}: {e}", dll.display()))?
            .with_name("local-code-rag")
            .build()
            .context("creating ONNX Runtime environment")?;
        let state = State { env, registered: Mutex::new(Vec::new()), dll_dir: Some(dir) };
        if let Err(e) = register_ready(&state) {
            tracing::warn!("Windows ML provider registration skipped: {e:#}");
        }
        Ok(state)
    }
    #[cfg(not(feature = "winml"))]
    {
        let env = ort::init().with_name("local-code-rag").build().context("creating ONNX Runtime environment")?;
        Ok(State { env, registered: Mutex::new(Vec::new()), dll_dir: None })
    }
}

/// Initialise ORT once per process. Safe to call repeatedly.
pub fn init() -> Result<()> {
    state().map(|_| ())
}

fn state() -> Result<&'static State> {
    STATE
        .get_or_init(|| build_state().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

pub fn environment() -> Result<Environment> {
    Ok(state()?.env.clone())
}

/// Register every certified Windows ML provider that is already installed.
#[cfg(feature = "winml")]
fn register_ready(state: &State) -> Result<Vec<String>> {
    let dir = state.dll_dir.as_ref().expect("winml has a dll dir");
    let mut newly = Vec::new();
    for p in crate::winml::list_providers(dir)? {
        if !p.certified || p.ready_state != crate::winml::ReadyState::Ready {
            continue;
        }
        let Some(lib) = p.library_path.as_ref() else { continue };
        let mut reg = state.registered.lock().unwrap();
        if reg.iter().any(|n| n == &p.name) {
            continue;
        }
        match state.env.register_ep_library(&p.name, lib) {
            Ok(()) => {
                tracing::info!("registered Windows ML provider {} {} ({})", p.name, p.version, lib.display());
                reg.push(p.name.clone());
                newly.push(p.name);
            }
            Err(e) => tracing::warn!("could not register {}: {e}", p.name),
        }
    }
    Ok(newly)
}

/// Re-scan Windows ML (after an install) and register newly ready providers.
/// Cached models are dropped so the next load picks the new device.
pub fn refresh_providers() -> Result<Vec<String>> {
    #[cfg(feature = "winml")]
    {
        let newly = register_ready(state()?)?;
        if !newly.is_empty() {
            crate::embedder::clear_caches();
        }
        Ok(newly)
    }
    #[cfg(not(feature = "winml"))]
    Ok(Vec::new())
}

/// Execution providers Windows ML offers for this device (empty for bundled builds).
#[cfg(feature = "winml")]
pub fn winml_providers() -> Result<Vec<crate::winml::Provider>> {
    let st = state()?;
    crate::winml::list_providers(st.dll_dir.as_ref().expect("winml has a dll dir"))
}

/// Download/install a Windows ML provider (blocking; run off the async runtime),
/// then register it with ORT and drop cached sessions.
#[cfg(feature = "winml")]
pub fn install_winml_provider(
    name: &str,
    cancel: &std::sync::atomic::AtomicBool,
    on_progress: &(dyn Fn(f64) + Sync),
) -> Result<Vec<String>> {
    let st = state()?;
    crate::winml::ensure_ready(st.dll_dir.as_ref().expect("winml has a dll dir"), name, cancel, on_progress)?;
    refresh_providers()
}

pub fn info() -> Result<RuntimeInfo> {
    let st = state()?;
    let devices: Vec<DeviceInfo> = st
        .env
        .devices()
        .map(|d| {
            let hw = d.hardware_device();
            DeviceInfo {
                ep: d.ep().unwrap_or("?").to_string(),
                ep_vendor: d.ep_vendor().unwrap_or("?").to_string(),
                kind: format!("{:?}", hw.ty()).to_ascii_lowercase(),
                vendor: hw.vendor().unwrap_or("?").to_string(),
            }
        })
        .collect();
    let vendor_gpu = devices.iter().any(|d| d.kind == "gpu" && !is_builtin_ep(&d.ep));
    Ok(RuntimeInfo {
        flavour: FLAVOUR,
        dll_dir: st.dll_dir.as_ref().map(|d| d.display().to_string()),
        registered_providers: st.registered.lock().unwrap().clone(),
        devices,
        vendor_gpu,
    })
}

/// Whether a GPU execution path may exist. Windows ML builds know their
/// devices (DirectML lists the GPU); bundled builds can only tell whether CUDA
/// support was compiled in (CUDA itself may still be missing at runtime, in
/// which case FP16 runs on CPU, slightly slower).
pub fn gpu_available() -> bool {
    #[cfg(feature = "winml")]
    {
        info().map(|i| i.devices.iter().any(|d| d.kind == "gpu")).unwrap_or(false)
    }
    #[cfg(not(feature = "winml"))]
    {
        cfg!(feature = "cuda")
    }
}

fn is_builtin_ep(ep: &str) -> bool {
    matches!(ep, "CPUExecutionProvider" | "DmlExecutionProvider")
}

/// How a session should pick its device: explicit EP list and/or an ORT
/// device-selection policy.
pub fn session_plan(ep: ExecutionProvider) -> (Vec<ExecutionProviderDispatch>, Option<AutoDevicePolicy>) {
    #[cfg(feature = "winml")]
    {
        let vendor_gpu = info().map(|i| i.vendor_gpu).unwrap_or(false);
        match ep {
            ExecutionProvider::Cpu => (vec![], Some(AutoDevicePolicy::PreferCPU)),
            ExecutionProvider::Npu => (vec![], Some(AutoDevicePolicy::PreferNPU)),
            ExecutionProvider::Directml => (vec![ort::ep::DirectML::default().build()], None),
            // Auto / Gpu / Cuda: a vendor EP from Windows ML if one is registered, else DirectML.
            _ if vendor_gpu => (vec![], Some(AutoDevicePolicy::PreferGPU)),
            _ => (vec![ort::ep::DirectML::default().build()], None),
        }
    }
    #[cfg(not(feature = "winml"))]
    {
        #[allow(unused_mut)]
        let mut v: Vec<ExecutionProviderDispatch> = Vec::new();
        #[cfg(feature = "cuda")]
        if matches!(ep, ExecutionProvider::Auto | ExecutionProvider::Cuda | ExecutionProvider::Gpu) {
            v.push(ort::ep::CUDA::default().build());
        }
        if v.is_empty() && !matches!(ep, ExecutionProvider::Cpu | ExecutionProvider::Auto) {
            tracing::warn!("execution provider {ep:?} is not available in this build; using CPU");
        }
        // Explicit policy: ORT's implicit default can delegate to unexpected devices.
        let policy = v.is_empty().then_some(AutoDevicePolicy::PreferCPU);
        (v, policy)
    }
}
