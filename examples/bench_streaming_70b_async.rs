//! Async H2D streaming bench — same prompt + token budget as
//! `bench_streaming_70b.rs` so wall-time is directly comparable to
//! the synchronous F16 row.
//!
//! Difference: `LoadOptions::async_h2d = true` switches StreamingLlama's
//! per-layer weight transfer from the synchronous mmap path to the
//! double-buffered `AsyncWeightLoader` (see `src/async_upload.rs`),
//! overlapping the next layer's H2D with the current layer's compute.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rustyllm::inference::LoadOptions;
use rustyllm::StreamingLlama;

const MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
const PROMPT: &str = "What is the capital of the United States, and what river runs through it?";
const MAX_NEW_TOKENS: usize = 16;
const LAYER_SHARDS_DIR: &str = "/mnt/4t/cache/airllm_splits";

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
        layer_shards_dir: Some(PathBuf::from(LAYER_SHARDS_DIR)),
        async_h2d: true,
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

    let t_load = Instant::now();
    println!("Loading {MODEL} (async H2D streaming, shards at {LAYER_SHARDS_DIR})...");
    let llama = StreamingLlama::from_pretrained(MODEL, opts)?;
    println!("  loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    println!("Generating {MAX_NEW_TOKENS} tokens...");
    let t_gen = Instant::now();
    let output = llama.generate(PROMPT, MAX_NEW_TOKENS, 0.0, None)?;
    let wall = t_gen.elapsed().as_secs_f64();
    let tps = MAX_NEW_TOKENS as f64 / wall;

    stop.store(true, Ordering::Relaxed);
    sampler.join().ok();
    let peak_mib = peak.load(Ordering::Relaxed);

    println!("\n=== RESULT ===");
    println!("PROMPT: {PROMPT}");
    println!("OUTPUT: {output}");
    println!("wall:        {wall:.1}s for {MAX_NEW_TOKENS} tokens");
    println!("tok/s:       {tps:.3}");
    println!("peak host:   {peak_mib} MiB");
    Ok(())
}
