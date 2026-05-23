//! Minimal CLI for one-shot generation via StreamingLlama.
//!
//! Output goes to stdout as JSON: {"tokens":[...], "text":"..."} for
//! easy programmatic consumption and token-level diffing (more
//! sensitive than decoded text, which can hide a single-token drift
//! inside a longer
//! decode).

use std::path::PathBuf;

use clap::Parser;
use rustyllm::inference::LoadOptions;
use rustyllm::StreamingLlama;

#[derive(Parser, Debug)]
struct Args {
    /// HuggingFace repo id or local directory.
    #[arg(long)]
    model: String,
    /// Prompt to generate from.
    #[arg(long)]
    prompt: String,
    /// Maximum new tokens to generate.
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    /// Optional top-p.
    #[arg(long)]
    top_p: Option<f64>,
    /// Reuse pre-split per-layer shards from this directory (e.g.,
    /// /mnt/caching/airllm/layer_shards). Skips re-download/re-shard.
    #[arg(long)]
    shards_dir: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let opts = LoadOptions {
        layer_shards_dir: args.shards_dir,
        ..LoadOptions::default()
    };

    let model = StreamingLlama::from_pretrained(&args.model, opts)?;
    let text = model.generate(
        &args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )?;

    // Emit decoded text only for now; token-sequence comparison would
    // need surfacing the post-generation token list from generate(),
    // which is a minor API addition we can do when needed.
    println!("{}", serde_json::json!({ "text": text }));
    Ok(())
}
