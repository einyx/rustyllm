//! Q4-quantized streaming bench. Same prompt + token budget as
//! `bench_streaming_70b.rs` so wall-time is directly comparable to the
//! F16 row in `bench_70b_apples_to_oranges.md`.
//!
//! Q4_K weights are ~4x smaller than F16, so per-token PCIe traffic
//! drops from ~140 GB to ~35 GB. The QMatMul kernel dequantizes on
//! device, keeping the compute work similar.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rustyllm::inference::LoadOptions;
use rustyllm::{ensure_q4_shards, PrepareOptions, Quantization, StreamingLlamaQuantized};

fn quantization() -> Quantization {
    match std::env::var("RUSTYLLM_QUANT").as_deref() {
        Ok("q2k") | Ok("q2_k") => Quantization::Q2K,
        Ok("q3k") | Ok("q3_k") => Quantization::Q3K,
        Ok("q4_0") => Quantization::Q4_0,
        Ok("q5k") | Ok("q5_k") => Quantization::Q5K,
        Ok("q6k") | Ok("q6_k") => Quantization::Q6K,
        Ok("q8_0") => Quantization::Q8_0,
        _ => Quantization::Q4K,
    }
}

const DEFAULT_MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
fn model() -> String {
    std::env::var("RUSTYLLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}
const DEFAULT_PROMPT: &str = "What is the capital of the United States, and what river runs through it?";
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
fn prompt() -> String {
    std::env::var("RUSTYLLM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string())
}
fn max_new_tokens() -> usize {
    std::env::var("RUSTYLLM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}
fn shards_dir_path() -> String {
    if let Ok(v) = std::env::var("RUSTYLLM_Q4_SHARDS_DIR") {
        return v;
    }
    let base = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    // Per-model + per-quant subdir so different configs don't clobber each other's shards.
    let model_slug = model().replace('/', "_");
    let qslug = format!("{:?}", quantization()).to_lowercase();
    format!("{base}/.cache/rustyllm/q4_shards/{model_slug}_{qslug}")
}

fn vm_rss_mib() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kib / 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        // SAFETY: getrusage is async-signal-safe and always succeeds for RUSAGE_SELF.
        // ru_maxrss on macOS is in bytes (unlike Linux where it's KiB).
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret == 0 {
            Some(usage.ru_maxrss as u64 / (1024 * 1024))
        } else {
            None
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let opts = LoadOptions {
        layer_shards_dir: Some(PathBuf::from(shards_dir_path())),
        ..LoadOptions::default()
    };

    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let peak = peak.clone();
        let stop = stop.clone();
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some(rss) = vm_rss_mib() {
                    peak.fetch_max(rss, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(200));
            }
        })
    };

    let dir = shards_dir_path();
    let prompt_str = prompt();
    let max_tokens = max_new_tokens();

    let model_id = model();

    // Build quantized shards if not already present (download → shard → quantize).
    let prep_opts = PrepareOptions {
        quantization: quantization(),
        ..PrepareOptions::default()
    };
    let q4_dir = ensure_q4_shards(&model_id, Path::new(&dir), prep_opts)?;

    let t_load = Instant::now();
    println!("Loading {model_id} (Q4 streaming, shards at {dir})...");
    let llama = StreamingLlamaQuantized::from_pretrained(&model_id, &q4_dir, opts)?;
    println!("  loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    println!("Generating {max_tokens} tokens...");
    let t_gen = Instant::now();
    let output = llama.generate(prompt_str.as_str(), max_tokens, 0.0, None)?;
    let wall = t_gen.elapsed().as_secs_f64();
    let tps = max_tokens as f64 / wall;

    stop.store(true, Ordering::Relaxed);
    sampler.join().ok();
    let peak_mib = peak.load(Ordering::Relaxed);

    println!("\n=== RESULT ===");
    println!("PROMPT: {prompt_str}");
    println!("OUTPUT: {output}");
    println!("wall:        {wall:.1}s for {max_tokens} tokens");
    println!("tok/s:       {tps:.3}");
    println!("peak host:   {peak_mib} MiB");
    Ok(())
}
