//! CLI: split a HuggingFace model into per-layer safetensor shards.

use std::path::PathBuf;

use clap::Parser;
use rustyllm::shard::{shard_model, ShardConfig};

#[derive(Parser)]
#[command(name = "rustyllm-shard")]
#[command(about = "Split HF Llama-style checkpoints into per-layer shards")]
struct Args {
    /// Local model directory or HuggingFace repo id (must be cached locally)
    model: String,

    /// Optional output directory for shards (default: <model>/splitted_model)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        anyhow::bail!(
            "model path {} does not exist; download the model first or pass a local path",
            model_dir.display()
        );
    }
    let paths = shard_model(&model_dir, args.output.as_deref(), &ShardConfig::default())?;
    println!("shards written to {}", paths.root.display());
    println!("layers: {}", paths.layers.len());
    Ok(())
}
