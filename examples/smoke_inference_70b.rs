//! End-to-end smoke test: load a 70B-class model and generate a few tokens.

use rustyllm::inference::{LayeredLlama, LoadOptions};

const MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
const PROMPT: &str = "What is the capital of United States?";
const MAX_NEW_TOKENS: usize = 8;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("Loading {MODEL}...");
    let llama = LayeredLlama::from_pretrained(MODEL, LoadOptions::default())?;

    println!("Generating...");
    let output = llama.generate(PROMPT, MAX_NEW_TOKENS, 0.0, None)?;

    println!("PROMPT: {PROMPT}");
    println!("OUTPUT: {output}");
    Ok(())
}
