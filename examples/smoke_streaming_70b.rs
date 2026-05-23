//! Per-layer streaming smoke test on a 70B model.
//!
//! Reuses pre-split shards under the configured directory so no
//! re-download or re-shard is needed.

use std::path::PathBuf;

use rustyllm::inference::LoadOptions;
use rustyllm::StreamingLlama;

const MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
const PROMPT: &str = "What is the capital of United States?";
const MAX_NEW_TOKENS: usize = 2;
const LAYER_SHARDS_DIR: &str = "/mnt/4t/cache/airllm_splits";

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let opts = LoadOptions {
        layer_shards_dir: Some(PathBuf::from(LAYER_SHARDS_DIR)),
        ..LoadOptions::default()
    };

    println!("Loading {MODEL} (streaming, shards at {LAYER_SHARDS_DIR})...");
    let llama = StreamingLlama::from_pretrained(MODEL, opts)?;

    println!("Generating...");
    let output = llama.generate(PROMPT, MAX_NEW_TOKENS, 0.0, None)?;

    println!("PROMPT: {PROMPT}");
    println!("OUTPUT: {output}");
    Ok(())
}
