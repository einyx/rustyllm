//! Re-quantize existing per-layer GGUF shards into a mixed-precision
//! layout: attention weights at one quant (default Q4K), MLP weights
//! at another (default Q2K). Reads from `RUSTYLLM_SRC_DIR`, writes to
//! `RUSTYLLM_DST_DIR`. Idempotent — skips shards whose output exists.
//!
//! Usage:
//!   RUSTYLLM_SRC_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-70B-instruct_q4k \
//!   RUSTYLLM_DST_DIR=~/.cache/rustyllm/q4_shards/garage-bAInd_Platypus2-70B-instruct_mixed \
//!     cargo run --release --example requantize_mixed

use std::path::PathBuf;
use std::time::Instant;

use rustyllm::{requantize_shard_mixed, Quantization};

fn parse_quant(s: &str) -> Quantization {
    match s.to_ascii_lowercase().as_str() {
        "q2k" | "q2_k" => Quantization::Q2K,
        "q3k" | "q3_k" => Quantization::Q3K,
        "q4_0" => Quantization::Q4_0,
        "q5k" | "q5_k" => Quantization::Q5K,
        "q6k" | "q6_k" => Quantization::Q6K,
        "q8_0" => Quantization::Q8_0,
        _ => Quantization::Q4K,
    }
}

fn main() -> anyhow::Result<()> {
    let src = PathBuf::from(
        std::env::var("RUSTYLLM_SRC_DIR")
            .map_err(|_| anyhow::anyhow!("set RUSTYLLM_SRC_DIR"))?,
    );
    let dst = PathBuf::from(
        std::env::var("RUSTYLLM_DST_DIR")
            .map_err(|_| anyhow::anyhow!("set RUSTYLLM_DST_DIR"))?,
    );
    let attn = parse_quant(&std::env::var("RUSTYLLM_ATTN_QUANT").unwrap_or_else(|_| "q4k".into()));
    let mlp = parse_quant(&std::env::var("RUSTYLLM_MLP_QUANT").unwrap_or_else(|_| "q2k".into()));
    let other = parse_quant(&std::env::var("RUSTYLLM_OTHER_QUANT").unwrap_or_else(|_| "q4k".into()));

    std::fs::create_dir_all(&dst)?;
    eprintln!(
        "[requant] {} → {} (attn={:?}, mlp={:?}, other={:?})",
        src.display(),
        dst.display(),
        attn,
        mlp,
        other
    );

    let mut entries: Vec<_> = std::fs::read_dir(&src)?
        .flatten()
        .filter(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let t_start = Instant::now();
    let (mut total_q, mut total_f16, mut total_t) = (0usize, 0usize, 0usize);
    for (i, entry) in entries.iter().enumerate() {
        let src_path = entry.path();
        let stem = src_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("bad name"))?;
        let dst_path = dst.join(format!("{stem}.gguf"));
        let t0 = Instant::now();
        let r = requantize_shard_mixed(&src_path, &dst_path, attn, mlp, other)?;
        let took = t0.elapsed().as_secs_f64();
        eprintln!(
            "[requant] {}/{} {} ({} tensors, {} q + {} f16, {:.1}s)",
            i + 1,
            entries.len(),
            stem,
            r.total,
            r.quantized,
            r.kept_f16,
            took
        );
        total_t += r.total;
        total_q += r.quantized;
        total_f16 += r.kept_f16;
    }
    let wall = t_start.elapsed().as_secs_f64();
    eprintln!(
        "[requant] DONE: {} tensors ({} q + {} f16) across {} shards in {:.1}s",
        total_t,
        total_q,
        total_f16,
        entries.len(),
        wall
    );
    Ok(())
}
