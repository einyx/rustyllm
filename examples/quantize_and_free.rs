//! Quantize existing F16 shards in-place, deleting each F16 as soon as
//! its GGUF lands on disk. Use when disk pressure prevents holding the
//! full F16 shard set + raw weights + GGUF outputs at once.

use std::path::PathBuf;

use rustyllm::{quantize_and_delete_f16, Quantization};

fn main() -> anyhow::Result<()> {
    let home = std::env::var("HOME")?;
    let f16_dir = std::env::var("RUSTYLLM_F16_SHARDS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{home}/.cache/rustyllm/f16_splits/splitted_model")));
    let q4_dir = std::env::var("RUSTYLLM_Q4_SHARDS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{home}/.cache/rustyllm/q4_shards")));

    eprintln!("[quantize+free] {} → {}", f16_dir.display(), q4_dir.display());
    let n = quantize_and_delete_f16(&f16_dir, &q4_dir, Quantization::Q4K)?;
    eprintln!("[quantize+free] Quantized {n} shards.");
    Ok(())
}
