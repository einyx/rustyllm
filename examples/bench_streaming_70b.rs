//! Streaming-70B wall-time bench: same prompt + token budget across
//! runs, so output rows are directly comparable.

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

/// Resident set size in MiB from /proc/self/status (Linux). Returns
/// `None` on non-Linux or if the format is unexpected.
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
        async_h2d: std::env::var("RUSTYLLM_ASYNC_H2D")
            .ok()
            .as_deref()
            == Some("1"),
        ..LoadOptions::default()
    };

    // Background RSS sampler. Polls every 200ms — adds ~0 overhead and
    // captures peak host memory even between candle allocator high
    // watermarks.
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
    println!("Loading {MODEL} (streaming, shards at {LAYER_SHARDS_DIR})...");
    let mut llama = StreamingLlama::from_pretrained(MODEL, opts)?;
    if let Ok(qdir) = std::env::var("RUSTYLLM_Q_STREAM_DIR") {
        println!("  mixed-precision streaming via {qdir}");
        llama.enable_quantized_streaming(&qdir)?;
    }
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
