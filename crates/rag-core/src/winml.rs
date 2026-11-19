//! Windows ML execution-provider catalog (flat C API from
//! `Microsoft.Windows.AI.MachineLearning.dll`, NuGet `Microsoft.Windows.AI.MachineLearning`).
//!
//! Windows ML downloads vendor execution providers (NVIDIA TensorRT-RTX, AMD, Intel
//! OpenVINO, Qualcomm QNN, ...) on demand and hands back the path of each EP's
//! plugin library, which is then registered with our ONNX Runtime environment.
//!
//! Bindings mirror `WinMLEpCatalog.h` / `WinMLAsync.h` from package 2.5.96.

use std::ffi::{c_char, c_void, CStr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde::Serialize;

pub const WINML_DLL: &str = "Microsoft.Windows.AI.MachineLearning.dll";

type HRESULT = i32;
type BOOL = i32;
type CatalogHandle = *mut c_void;
type EpHandle = *mut c_void;

const E_PENDING: HRESULT = 0x8000_000Au32 as i32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum ReadyStateC {
    Ready = 0,
    NotReady = 1,
    NotPresent = 2,
}

#[repr(C)]
struct WinMLEpInfo {
    name: *const c_char,
    version: *const c_char,
    package_family_name: *const c_char,
    library_path: *const c_char,
    package_root_path: *const c_char,
    ready_state: i32,
    certification: i32,
}

type EnumCallback = unsafe extern "system" fn(EpHandle, *const WinMLEpInfo, *mut c_void) -> BOOL;
type AsyncCompletion = unsafe extern "system" fn(*mut WinMLAsyncBlock);
type AsyncProgress = unsafe extern "system" fn(*mut WinMLAsyncBlock, f64);

#[repr(C)]
struct WinMLAsyncBlock {
    context: *mut c_void,
    callback: Option<AsyncCompletion>,
    progress: Option<AsyncProgress>,
}

struct Api {
    _lib: Library,
    catalog_create: unsafe extern "system" fn(*mut CatalogHandle) -> HRESULT,
    catalog_release: unsafe extern "system" fn(CatalogHandle),
    catalog_enum: unsafe extern "system" fn(CatalogHandle, EnumCallback, *mut c_void) -> HRESULT,
    ensure_ready_async: unsafe extern "system" fn(EpHandle, *mut WinMLAsyncBlock) -> HRESULT,
    async_get_status: unsafe extern "system" fn(*mut WinMLAsyncBlock, BOOL) -> HRESULT,
    async_cancel: unsafe extern "system" fn(*mut WinMLAsyncBlock) -> HRESULT,
    async_close: unsafe extern "system" fn(*mut WinMLAsyncBlock),
    get_library_path_size: unsafe extern "system" fn(EpHandle, *mut usize) -> HRESULT,
    get_library_path: unsafe extern "system" fn(EpHandle, usize, *mut c_char, *mut usize) -> HRESULT,
}

// The function pointers are plain C entry points; the library stays loaded for
// the life of the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn hr(code: HRESULT, what: &str) -> Result<()> {
    if code >= 0 {
        Ok(())
    } else {
        bail!("{what} failed (HRESULT 0x{:08X})", code as u32)
    }
}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

/// Load `Microsoft.Windows.AI.MachineLearning.dll` from `dir` (once per process).
fn api_in(dir: &Path) -> Result<&'static Api> {
    API.get_or_init(|| unsafe {
        let path = dir.join(WINML_DLL);
        let lib = Library::new(&path).map_err(|e| format!("loading {}: {e}", path.display()))?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get($name).map_err(|e| format!("{}: {e}", String::from_utf8_lossy($name)))?
            };
        }
        Ok(Api {
            catalog_create: sym!(b"WinMLEpCatalogCreate"),
            catalog_release: sym!(b"WinMLEpCatalogRelease"),
            catalog_enum: sym!(b"WinMLEpCatalogEnumProviders"),
            ensure_ready_async: sym!(b"WinMLEpEnsureReadyAsync"),
            async_get_status: sym!(b"WinMLAsyncGetStatus"),
            async_cancel: sym!(b"WinMLAsyncCancel"),
            async_close: sym!(b"WinMLAsyncClose"),
            get_library_path_size: sym!(b"WinMLEpGetLibraryPathSize"),
            get_library_path: sym!(b"WinMLEpGetLibraryPath"),
            _lib: lib,
        })
    })
    .as_ref()
    .map_err(|e| anyhow!("Windows ML unavailable: {e}"))
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadyState {
    /// Installed and usable.
    Ready,
    /// Installed but needs a (re)download/update before use.
    NotReady,
    /// Available for this device but not downloaded yet.
    NotPresent,
}

#[derive(Debug, Clone, Serialize)]
pub struct Provider {
    pub name: String,
    pub version: String,
    pub ready_state: ReadyState,
    pub certified: bool,
    /// Plugin library to register with ONNX Runtime (only meaningful when ready).
    pub library_path: Option<PathBuf>,
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// RAII catalog; EP handles stay valid while it lives.
struct Catalog {
    api: &'static Api,
    handle: CatalogHandle,
}

impl Drop for Catalog {
    fn drop(&mut self) {
        unsafe { (self.api.catalog_release)(self.handle) }
    }
}

impl Catalog {
    fn open(dir: &Path) -> Result<Self> {
        let api = api_in(dir)?;
        let mut handle: CatalogHandle = std::ptr::null_mut();
        hr(unsafe { (api.catalog_create)(&mut handle) }, "WinMLEpCatalogCreate")?;
        if handle.is_null() {
            bail!("WinMLEpCatalogCreate returned a null catalog");
        }
        Ok(Self { api, handle })
    }

    fn providers(&self) -> Result<Vec<(EpHandle, Provider)>> {
        unsafe extern "system" fn cb(ep: EpHandle, info: *const WinMLEpInfo, ctx: *mut c_void) -> BOOL {
            let out = unsafe { &mut *(ctx as *mut Vec<(EpHandle, Provider)>) };
            if let Some(info) = unsafe { info.as_ref() } {
                let ready_state = match info.ready_state {
                    0 => ReadyState::Ready,
                    1 => ReadyState::NotReady,
                    _ => ReadyState::NotPresent,
                };
                let lib = cstr(info.library_path);
                out.push((
                    ep,
                    Provider {
                        name: cstr(info.name),
                        version: cstr(info.version),
                        ready_state,
                        certified: info.certification == 1,
                        library_path: (!lib.is_empty()).then(|| PathBuf::from(lib)),
                    },
                ));
            }
            1 // TRUE: continue enumeration
        }
        let mut out: Vec<(EpHandle, Provider)> = Vec::new();
        hr(
            unsafe { (self.api.catalog_enum)(self.handle, cb, &mut out as *mut _ as *mut c_void) },
            "WinMLEpCatalogEnumProviders",
        )?;
        // Re-query library paths through the API (the info struct can be empty
        // until the package is installed).
        for (ep, p) in &mut out {
            if p.ready_state == ReadyState::Ready {
                if let Ok(path) = self.library_path(*ep) {
                    p.library_path = Some(path);
                }
            }
        }
        Ok(out)
    }

    fn library_path(&self, ep: EpHandle) -> Result<PathBuf> {
        let mut size = 0usize;
        hr(unsafe { (self.api.get_library_path_size)(ep, &mut size) }, "WinMLEpGetLibraryPathSize")?;
        if size == 0 {
            bail!("empty EP library path");
        }
        let mut buf = vec![0u8; size];
        hr(
            unsafe { (self.api.get_library_path)(ep, size, buf.as_mut_ptr() as *mut c_char, std::ptr::null_mut()) },
            "WinMLEpGetLibraryPath",
        )?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(PathBuf::from(String::from_utf8_lossy(&buf[..end]).into_owned()))
    }
}

/// All execution providers Windows ML offers for this device.
pub fn list_providers(dll_dir: &Path) -> Result<Vec<Provider>> {
    Ok(Catalog::open(dll_dir)?.providers()?.into_iter().map(|(_, p)| p).collect())
}

struct ProgressCtx<'a> {
    on_progress: &'a (dyn Fn(f64) + Sync),
}

/// Download/install one provider (blocking). `on_progress` receives 0..=100.
pub fn ensure_ready(
    dll_dir: &Path,
    provider: &str,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(f64) + Sync),
) -> Result<Provider> {
    let catalog = Catalog::open(dll_dir)?;
    let (ep, info) = catalog
        .providers()?
        .into_iter()
        .find(|(_, p)| p.name.eq_ignore_ascii_case(provider))
        .ok_or_else(|| anyhow!("Windows ML does not offer provider '{provider}' on this device"))?;
    if info.ready_state == ReadyState::Ready {
        return Ok(info);
    }

    unsafe extern "system" fn progress(block: *mut WinMLAsyncBlock, value: f64) {
        if let Some(block) = unsafe { block.as_ref() } {
            if let Some(ctx) = unsafe { (block.context as *const ProgressCtx).as_ref() } {
                (ctx.on_progress)(value);
            }
        }
    }

    let ctx = ProgressCtx { on_progress };
    let mut block = WinMLAsyncBlock {
        context: &ctx as *const ProgressCtx as *mut c_void,
        callback: None,
        progress: Some(progress),
    };
    let api = catalog.api;
    hr(unsafe { (api.ensure_ready_async)(ep, &mut block) }, "WinMLEpEnsureReadyAsync")?;

    // Poll so cancellation stays responsive; `block` and `ctx` outlive the operation.
    let result = loop {
        let status = unsafe { (api.async_get_status)(&mut block, 0) };
        if status != E_PENDING {
            break hr(status, &format!("installing {provider}"));
        }
        if cancel.load(Ordering::Relaxed) {
            unsafe { (api.async_cancel)(&mut block) };
            let _ = unsafe { (api.async_get_status)(&mut block, 1) };
            break Err(anyhow!(crate::Cancelled));
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    unsafe { (api.async_close)(&mut block) };
    result?;

    let path = catalog.library_path(ep).context("provider installed but has no library path")?;
    Ok(Provider { ready_state: ReadyState::Ready, library_path: Some(path), ..info })
}
