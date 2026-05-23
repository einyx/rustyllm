//! CLI: quantize per-layer F16 safetensors shards into per-layer GGUF.
//!
//! Reads each `*.safetensors` in the source directory, quantizes its
//! bulk weight matrices (per `quantize::should_quantize`), keeps norms
//! and embeddings at F16, writes a paired `*.gguf` to the destination
//! directory. Used to prepare the inputs for `StreamingLlamaQuantized`.
//!
//! Usage:
//!   cargo run --release --bin rustyllm-quantize-shards -- \
//!     --src /mnt/4t/cache/airllm_splits/splitted_model \
//!     --dst /mnt/4t/cache/airllm_splits_q4_0 \
//!     --format q4_0
//!
//! Expect ~5-15s per layer on CPU; for an 80-layer model that's ~10-20
//! min total. Output is ~4x smaller than the input.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rustyllm::{quantize_split_directory, Quantization};

#[derive(Parser, Debug)]
struct Args {
    /// Source directory holding per-layer F16 safetensors shards.
    #[arg(long)]
    src: PathBuf,
    /// Destination directory for per-layer GGUF outputs.
    #[arg(long)]
    dst: PathBuf,
    /// Quantization format: q4_0 (best ratio), q4_k (best quality), q8_0 (safest).
    #[arg(long, default_value = "q4_0")]
    format: String,
}

fn parse_format(s: &str) -> anyhow::Result<Quantization> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q2_k" | "q2k" => Quantization::Q2K,
        "q3_k" | "q3k" => Quantization::Q3K,
        "q4_0" => Quantization::Q4_0,
        "q4_k" | "q4k" => Quantization::Q4K,
        "q5_k" | "q5k" => Quantization::Q5K,
        "q6_k" | "q6k" => Quantization::Q6K,
        "q8_0" => Quantization::Q8_0,
        other => anyhow::bail!("unknown format `{other}`; want q2_k|q3_k|q4_0|q4_k|q5_k|q6_k|q8_0"),
    })
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let q = parse_format(&args.format)?;

    eprintln!(
        "[quantize-shards] {} → {} (format: {:?})",
        args.src.display(),
        args.dst.display(),
        q
    );
    std::fs::create_dir_all(&args.dst)?;

    let t0 = Instant::now();
    let reports = quantize_split_directory(&args.src, &args.dst, q)?;
    let wall = t0.elapsed().as_secs_f64();

    let total_tensors: usize = reports.iter().map(|r| r.total).sum();
    let quantized: usize = reports.iter().map(|r| r.quantized).sum();
    let kept_f16: usize = reports.iter().map(|r| r.kept_f16).sum();

    eprintln!(
        "[quantize-shards] done: {} shards, {} tensors ({} quantized, {} f16) in {:.1}s",
        reports.len(),
        total_tensors,
        quantized,
        kept_f16,
        wall,
    );
    Ok(())
}
