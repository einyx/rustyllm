//! Per-layer streaming smoke test: loads only one block onto the device
//! at a time. Verifies the streaming forward path end-to-end against a
//! 7B model. Use a 70B model to demonstrate fitting on small VRAM.

use rustyllm::inference::LoadOptions;
use rustyllm::StreamingLlama;

const MODEL: &str = "garage-bAInd/Platypus2-7B";
const PROMPT: &str = "What is the capital of United States?";
const MAX_NEW_TOKENS: usize = 8;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("Loading {MODEL} (streaming)...");
    let llama = StreamingLlama::from_pretrained(MODEL, LoadOptions::default())?;

    println!("Generating...");
    let output = llama.generate(PROMPT, MAX_NEW_TOKENS, 0.0, None)?;

    println!("PROMPT: {PROMPT}");
    println!("OUTPUT: {output}");
    Ok(())
}
