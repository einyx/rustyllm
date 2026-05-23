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
use rustyllm::StreamingLlamaQuantized;

const MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
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
const Q4_SHARDS_DIR_DEFAULT: &str = "/mnt/4t/cache/airllm_splits_q4k";
fn shards_dir_path() -> String {
    std::env::var("RUSTYLLM_Q4_SHARDS_DIR").unwrap_or_else(|_| Q4_SHARDS_DIR_DEFAULT.to_string())
}

fn vm_rss_mib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kib / 1024);
        }
    }
    None
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
    let t_load = Instant::now();
    println!("Loading {MODEL} (Q4 streaming, shards at {dir})...");
    let llama = StreamingLlamaQuantized::from_pretrained(MODEL, Path::new(&dir), opts)?;
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
