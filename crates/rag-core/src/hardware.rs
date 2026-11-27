//! Hardware facts for the first-run setup: GPUs with dedicated VRAM (from the
//! display-adapter registry keys), system RAM, CPU threads, and a matching
//! model / precision / resource-mode recommendation. Read-only; no shell or WMI.

use serde::Serialize;

use crate::config::{Precision, ResourceMode};

#[derive(Debug, Clone, Serialize)]
pub struct Gpu {
    pub name: String,
    /// Dedicated video memory (0 when unknown).
    pub vram_gb: f32,
    /// Integrated GPU / APU (shares system RAM).
    pub integrated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hardware {
    pub cpu_threads: usize,
    pub ram_gb: f32,
    pub gpus: Vec<Gpu>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Discrete GPU with plenty of VRAM.
    Gpu,
    /// Discrete GPU with little VRAM (< 6 GB).
    SmallGpu,
    /// Integrated GPU / APU only (laptops, office PCs).
    Integrated,
    /// No usable GPU.
    Cpu,
}

#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub tier: Tier,
    pub model: String,
    pub precision: Precision,
    pub resource_mode: ResourceMode,
    /// One-line explanation for the UI.
    pub reason: String,
}

pub const MODEL_DEFAULT: &str = "jinaai/jina-embeddings-v2-base-code";
pub const MODEL_LIGHT: &str = "BAAI/bge-small-en-v1.5";

/// Pick model, precision and resource mode for the detected hardware.
pub fn recommend(hw: &Hardware) -> Recommendation {
    let best = hw.gpus.iter().filter(|g| !g.integrated).map(|g| g.vram_gb).fold(None, |a: Option<f32>, v| {
        Some(a.map_or(v, |a| a.max(v)))
    });
    let tier = match best {
        Some(v) if v >= 6.0 => Tier::Gpu,
        Some(_) => Tier::SmallGpu,
        None if hw.gpus.iter().any(|g| g.integrated) => Tier::Integrated,
        None => Tier::Cpu,
    };
    let (model, precision, resource_mode, reason) = match tier {
        Tier::Gpu => (
            MODEL_DEFAULT,
            Precision::Standard,
            ResourceMode::Auto,
            "Discrete GPU with enough memory: the code-trained default model at full quality and speed.",
        ),
        Tier::SmallGpu => (
            MODEL_DEFAULT,
            Precision::Standard,
            ResourceMode::Auto,
            "Discrete GPU: the code-trained default model fits; batches are sized to the GPU automatically.",
        ),
        Tier::Integrated => (
            MODEL_LIGHT,
            Precision::Standard,
            ResourceMode::Low,
            "Integrated graphics share memory with the system: a small, fast model and low-impact indexing keep the PC responsive.",
        ),
        Tier::Cpu => (
            MODEL_LIGHT,
            Precision::Int8,
            ResourceMode::Low,
            "No GPU acceleration: a small model in INT8 is the fastest option on the CPU.",
        ),
    };
    Recommendation { tier, model: model.into(), precision, resource_mode, reason: reason.into() }
}

/// Integrated when little dedicated VRAM is reported, or by well-known iGPU names.
#[cfg_attr(not(windows), allow(dead_code))]
fn is_integrated(name: &str, vram_gb: f32) -> bool {
    let n = name.to_ascii_lowercase();
    let igpu_name = n.contains("uhd graphics")
        || n.contains("iris")
        || n.contains("hd graphics")
        || (n.contains("intel") && n.contains("arc") && !n.contains("arc(tm) a") && !n.contains("arc a") && !n.contains(" b5") && !n.contains(" b7"))
        || n.trim_end().ends_with("radeon(tm) graphics")
        || n.trim_end().ends_with("radeon graphics")
        || n.contains("radeon 7")
        || n.contains("radeon 8")
        || n.contains("adreno");
    igpu_name || (vram_gb > 0.0 && vram_gb < 2.0)
}

#[cfg_attr(not(windows), allow(dead_code))]
fn skip_adapter(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("basic display") || n.contains("basic render") || n.contains("remote") || n.contains("virtual") || n.contains("indirect")
}

pub fn detect() -> Hardware {
    Hardware {
        cpu_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        ram_gb: ram_bytes() as f32 / (1u64 << 30) as f32,
        gpus: gpus(),
    }
}

#[cfg(windows)]
fn ram_bytes() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.ullTotalPhys } else { 0 }
}

#[cfg(not(windows))]
fn ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Display adapters from `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-…}\NNNN`
/// (`DriverDesc`, `HardwareInformation.qwMemorySize` / `.MemorySize`).
#[cfg(windows)]
fn gpus() -> Vec<Gpu> {
    use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_ANY};
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let read = |key: &[u16], value: &str| -> Option<Vec<u8>> {
        let v = wide(value);
        let mut size = 0u32;
        let mut ty = 0u32;
        let rc = unsafe {
            RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), v.as_ptr(), RRF_RT_ANY, &mut ty, std::ptr::null_mut(), &mut size)
        };
        if rc != 0 || size == 0 {
            return None;
        }
        let mut buf = vec![0u8; size as usize];
        let rc = unsafe {
            RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), v.as_ptr(), RRF_RT_ANY, &mut ty, buf.as_mut_ptr().cast(), &mut size)
        };
        (rc == 0).then(|| {
            buf.truncate(size as usize);
            buf
        })
    };
    let mut out: Vec<Gpu> = Vec::new();
    for i in 0..32 {
        let key = wide(&format!(
            "SYSTEM\\CurrentControlSet\\Control\\Class\\{{4d36e968-e325-11ce-bfc1-08002be10318}}\\{i:04}"
        ));
        let Some(name) = read(&key, "DriverDesc") else { continue };
        let name: String = String::from_utf16_lossy(
            &name.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&c| c != 0).collect::<Vec<_>>(),
        );
        if name.trim().is_empty() || skip_adapter(&name) || out.iter().any(|g| g.name == name) {
            continue;
        }
        let bytes = read(&key, "HardwareInformation.qwMemorySize")
            .or_else(|| read(&key, "HardwareInformation.MemorySize"))
            .map(|b| match b.len() {
                8.. => u64::from_le_bytes(b[..8].try_into().unwrap()),
                4..=7 => u32::from_le_bytes(b[..4].try_into().unwrap()) as u64,
                _ => 0,
            })
            .unwrap_or(0);
        let vram_gb = bytes as f32 / (1u64 << 30) as f32;
        out.push(Gpu { integrated: is_integrated(&name, vram_gb), name, vram_gb });
    }
    out
}

#[cfg(not(windows))]
fn gpus() -> Vec<Gpu> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(gpus: &[(&str, f32)]) -> Hardware {
        Hardware {
            cpu_threads: 16,
            ram_gb: 32.0,
            gpus: gpus
                .iter()
                .map(|(n, v)| Gpu { name: n.to_string(), vram_gb: *v, integrated: is_integrated(n, *v) })
                .collect(),
        }
    }

    #[test]
    fn tiers() {
        assert_eq!(recommend(&hw(&[("NVIDIA GeForce RTX 4080", 16.0)])).tier, Tier::Gpu);
        assert_eq!(recommend(&hw(&[("NVIDIA GeForce GTX 1650", 4.0)])).tier, Tier::SmallGpu);
        let laptop = recommend(&hw(&[("Intel(R) Arc(TM) Graphics", 0.125)]));
        assert_eq!(laptop.tier, Tier::Integrated);
        assert_eq!(laptop.model, MODEL_LIGHT);
        assert_eq!(laptop.resource_mode, ResourceMode::Low);
        // AMD APU with a large UMA frame buffer is still integrated.
        assert_eq!(recommend(&hw(&[("AMD Radeon(TM) Graphics", 4.0)])).tier, Tier::Integrated);
        // Laptop with iGPU + discrete GPU uses the discrete one.
        assert_eq!(recommend(&hw(&[("Intel(R) UHD Graphics", 0.1), ("NVIDIA GeForce RTX 4070 Laptop GPU", 8.0)])).tier, Tier::Gpu);
        assert_eq!(recommend(&hw(&[("Intel(R) Arc(TM) A770 Graphics", 16.0)])).tier, Tier::Gpu);
        let cpu = recommend(&hw(&[]));
        assert_eq!((cpu.tier, cpu.precision), (Tier::Cpu, Precision::Int8));
    }
}
