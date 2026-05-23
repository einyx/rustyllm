//! Speculative decoding bench: small draft model proposes K tokens at a
//! time, large target model verifies them in a single batched forward.
//!
//! Algorithm (Leviathan et al / Chen et al, with greedy verifier):
//!   1. Target's last sampled token = T (carried over from previous step).
//!   2. Draft autoregressively generates d1..dK from T, updating draft KV.
//!   3. Target runs forward on [T, d1, ..., dK] in a single call,
//!      producing logits at K+1 positions.
//!   4. For i in 0..K: if argmax(target_logits[i]) == d_{i+1}, accept.
//!      Otherwise stop at the first mismatch.
//!   5. Commit the accepted drafts + one bonus token = argmax(target_logits[j])
//!      at the rejection point.
//!   6. Roll back both caches to keep only the committed prefix.
//!
//! Best case (all K drafts match): K+1 tokens per target forward → up to
//! (K+1)x throughput. Realistic: 1.5-3x at K=4 with same-family draft on
//! predictable continuations.
//!
//! Target output is mathematically identical to greedy decoding without
//! speculation — the verifier is always the target model.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use candle_core::{IndexOp, Tensor, D};
use rustyllm::inference::LoadOptions;
use rustyllm::streaming_quantized::QuantStreamCache;
use rustyllm::StreamingLlamaQuantized;

const TARGET_MODEL: &str = "garage-bAInd/Platypus2-70B-instruct";
const DRAFT_MODEL: &str = "garage-bAInd/Platypus2-7B";
const TARGET_Q4_DIR_DEFAULT: &str = "/mnt/4t/cache/airllm_splits_q4k";
const DRAFT_Q4_DIR_DEFAULT: &str = "/mnt/4t/cache/platypus2_7b_q4k";
fn target_dir() -> String {
    std::env::var("RUSTYLLM_TARGET_DIR").unwrap_or_else(|_| TARGET_Q4_DIR_DEFAULT.to_string())
}
fn draft_dir() -> String {
    std::env::var("RUSTYLLM_DRAFT_DIR").unwrap_or_else(|_| DRAFT_Q4_DIR_DEFAULT.to_string())
}
const DEFAULT_PROMPT: &str = "What is the capital of the United States, and what river runs through it?";
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
fn prompt() -> String {
    std::env::var("RUSTYLLM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string())
}
fn max_new_tokens() -> usize {
    std::env::var("RUSTYLLM_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}
// Override via RUSTYLLM_SPEC_K=N. Default 2 because the verify forward at
// seq_len=K+1 has super-linear cost on this stack — empirically K=2 is
// the sweet spot across prompt types.
const DEFAULT_SPEC_K: usize = 2;
fn spec_k() -> usize {
    std::env::var("RUSTYLLM_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SPEC_K)
}

fn argmax_last(logits: &Tensor) -> Result<u32> {
    // logits shape: (vocab,) or (batch, seq, vocab) — caller already sliced.
    let id = logits.argmax(D::Minus1)?;
    Ok(id.to_scalar::<u32>().unwrap_or_else(|_| {
        id.to_vec1::<u32>().unwrap()[0]
    }))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let target_opts = LoadOptions {
        layer_shards_dir: Some(PathBuf::from("/mnt/4t/cache/airllm_splits")),
        ..LoadOptions::default()
    };
    let draft_opts = LoadOptions {
        layer_shards_dir: None,
        ..LoadOptions::default()
    };

    println!("Loading target {TARGET_MODEL} from {}...", target_dir());
    let t0 = Instant::now();
    let mut target = StreamingLlamaQuantized::from_pretrained(
        TARGET_MODEL,
        Path::new(target_dir().as_str()),
        target_opts,
    )?;
    // Target: 8 GPU layers (~4.8 GB) leaves room for draft fully on GPU.
    let target_gpu = std::env::var("RUSTYLLM_TARGET_GPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8usize);
    target.pin_resident_layers(target_gpu)?;
    target.pin_cpu_resident_layers(80 - target_gpu)?;
    println!("  target loaded in {:.1}s", t0.elapsed().as_secs_f64());

    println!("Loading draft {DRAFT_MODEL} from {}...", draft_dir());
    let t0 = Instant::now();
    let mut draft = StreamingLlamaQuantized::from_pretrained(
        DRAFT_MODEL,
        Path::new(draft_dir().as_str()),
        draft_opts,
    )?;
    // 7B has 32 layers. Pin all on GPU when there's headroom (~3.8 GB).
    let draft_gpu = std::env::var("RUSTYLLM_DRAFT_GPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);
    draft.pin_resident_layers(draft_gpu)?;
    if draft_gpu < 32 {
        draft.pin_cpu_resident_layers(32 - draft_gpu)?;
    }
    println!("  draft loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let prompt_str = prompt();
    let max_tokens = max_new_tokens();
    let tokenizer = target
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("target tokenizer missing"))?;
    let encoded = tokenizer.encode(prompt_str.as_str(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_len = tokens.len();

    let target_cfg = target.config().clone();
    let draft_cfg = draft.config().clone();
    let target_device = candle_core::Device::cuda_if_available(0)?;
    let mut t_cache = QuantStreamCache::new(
        true,
        candle_core::DType::F32,
        &target_cfg,
        &target_device,
    )?;
    let mut d_cache = QuantStreamCache::new(
        true,
        candle_core::DType::F32,
        &draft_cfg,
        &target_device,
    )?;

    // Prefill both models on the prompt (one forward each).
    println!("Prefilling prompt ({prompt_len} tokens)...");
    let prefill = Tensor::new(&tokens[..], &target_device)?.unsqueeze(0)?;
    let target_prefill_logits = target.forward(&prefill, 0, &mut t_cache)?;
    let _ = draft.forward(&prefill, 0, &mut d_cache)?;
    let mut index_pos = prompt_len;
    // First sampled token: greedy from the target's prompt-position logits.
    // forward returns logits at the LAST position only (see streaming_quantized
    // forward: `x = x.i((.., seq_len - 1, ..))` before lm_head).
    let mut last_token = argmax_last(&target_prefill_logits.i(0)?)?;
    tokens.push(last_token);

    let k_max = spec_k();
    let adaptive = std::env::var("RUSTYLLM_ADAPTIVE_K")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut k_now = k_max;
    // Sliding window of (accepted, attempted) over the last N rounds.
    let win_size = 4usize;
    let mut recent: std::collections::VecDeque<(usize, usize)> =
        std::collections::VecDeque::with_capacity(win_size);
    println!(
        "Generating up to {max_tokens} tokens with K={k_max} speculative (adaptive={adaptive})..."
    );
    print!("\n>>> ");
    // Stream the prompt's "last_token" so the user sees something before
    // the first target verify finishes. Decoded against the full token
    // history so SentencePiece word-boundary markers resolve correctly.
    let mut emitted_chars = 0usize;
    let first_chunk = tokenizer
        .decode(&tokens[prompt_len..], false)
        .unwrap_or_default();
    print!("{first_chunk}");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    emitted_chars = first_chunk.len();
    let t_gen = Instant::now();
    let mut accepted_total = 0usize;
    let mut target_forwards = 0usize;
    let mut draft_forwards = 0usize;
    let target_eos = target_cfg.eos_token_id.clone();

    while tokens.len() - prompt_len < max_tokens + 1 {
        // 1. Draft generates K candidate tokens autoregressively, each
        //    advancing d_cache by one position. Inputs to the draft are
        //    last_token, d1, d2, ..., d_{K-1}; outputs predict d1..dK.
        // When k_now == 0 we skip drafting entirely and fall back to a
        //    single-token target forward (true baseline).
        let mut drafts: Vec<u32> = Vec::with_capacity(k_now);
        let mut cur = last_token;
        for k in 0..k_now {
            let input = Tensor::new(&[cur], &target_device)?.unsqueeze(0)?;
            let logits = draft.forward(&input, index_pos + k, &mut d_cache)?;
            let tok = argmax_last(&logits.i(0)?)?;
            drafts.push(tok);
            cur = tok;
            draft_forwards += 1;
        }

        // 2. Target verifies in one forward over [last_token, d1..d_K]
        //    (K+1 tokens). target_logits[i] for i in 0..K predicts what
        //    target *would have* sampled given [last, d1..d_i], so we
        //    compare against drafts[i]. target_logits[K] is the bonus.
        let mut verify_input = vec![last_token];
        verify_input.extend_from_slice(&drafts);
        let verify_tensor =
            Tensor::new(&verify_input[..], &target_device)?.unsqueeze(0)?;
        let target_logits = target.forward_all(&verify_tensor, index_pos, &mut t_cache)?;
        target_forwards += 1;
        // Shape: (1, k_now+1, vocab) → slice [0] → (k_now+1, vocab).
        let logits_2d = target_logits.i(0)?;

        // 3. Find longest accepted prefix.
        let mut accepted = 0usize;
        let mut bonus = argmax_last(&logits_2d.i(k_now)?)?; // tentative
        for i in 0..k_now {
            let pred = argmax_last(&logits_2d.i(i)?)?;
            if pred == drafts[i] {
                accepted += 1;
            } else {
                bonus = pred;
                break;
            }
        }

        // 4. Commit drafts[0..accepted] + bonus. Stream tokens to stdout
        //    as they're committed. SentencePiece tokens carry word-
        //    boundary markers (▁ = space) that only make sense in
        //    context, so we decode the running sequence each time and
        //    print the diff vs what we've already emitted.
        use std::io::Write;
        for d in drafts[..accepted].iter().copied().chain(std::iter::once(bonus)) {
            tokens.push(d);
            let full = tokenizer
                .decode(&tokens[prompt_len..], false)
                .unwrap_or_default();
            if full.len() > emitted_chars {
                print!("{}", &full[emitted_chars..]);
                let _ = std::io::stdout().flush();
                emitted_chars = full.len();
            }
        }
        accepted_total += accepted;

        // 5. Cache bookkeeping.
        //   Target cache holds positions [0, index_pos+K+1) after the
        //   verify forward. Draft cache holds [0, index_pos+K) — we
        //   forwarded last_token + d1..d_{K-1} (K tokens), d_K was
        //   sampled but not fed back.
        //   Committed: accepted drafts + bonus = accepted+1 new tokens.
        //   Bonus is the new last_token (not yet forwarded). Caches
        //   should hold positions for everything BEFORE bonus, i.e.
        //   index_pos + accepted + 1 positions total.
        //
        //   Edge case: when accepted == K, draft cache is at K positions
        //   but we want K+1 — fill in d_K's KV with one extra forward
        //   so draft and target stay in sync for next round.
        if k_now > 0 && accepted == k_now {
            let last_d = drafts[k_now - 1];
            let input = Tensor::new(&[last_d], &target_device)?.unsqueeze(0)?;
            let _ = draft.forward(&input, index_pos + k_now, &mut d_cache)?;
            draft_forwards += 1;
        }
        let new_len = index_pos + accepted + 1;
        target.truncate_caches(&mut t_cache, new_len)?;
        draft.truncate_caches(&mut d_cache, new_len)?;

        // 6. Adaptive K: shrink fast when the draft isn't earning its
        // keep, grow slowly back. We monitor accept rate over a short
        // window; below ~40% the verify-forward cost at seq_len=K+1
        // exceeds what we save vs single-token baseline, so we drop K
        // (eventually to 0 = pure baseline). Above 70% with K<k_max we
        // grow K back.
        if adaptive {
            recent.push_back((accepted, k_now));
            if recent.len() > win_size {
                recent.pop_front();
            }
            let (acc_sum, att_sum) = recent
                .iter()
                .fold((0usize, 0usize), |(a, b), (x, y)| (a + x, b + y));
            let rate = if att_sum > 0 { acc_sum as f64 / att_sum as f64 } else { 1.0 };
            if recent.len() >= 2 && rate < 0.40 && k_now > 0 {
                k_now -= 1;
                recent.clear();
            } else if recent.len() >= win_size && rate > 0.70 && k_now < k_max {
                k_now += 1;
                recent.clear();
            }
        }

        index_pos = new_len;
        last_token = bonus;

        // EOS check.
        if let Some(eos) = target_eos.as_ref() {
            use candle_transformers::models::llama::LlamaEosToks;
            let stop = match eos {
                LlamaEosToks::Single(id) => bonus == *id,
                LlamaEosToks::Multiple(ids) => ids.contains(&bonus),
            };
            if stop {
                break;
            }
        }
    }

    let wall = t_gen.elapsed().as_secs_f64();
    let generated = tokens.len() - prompt_len;
    let text = tokenizer.decode(&tokens, true).map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("\n=== RESULT ===");
    println!("prompt_str: {prompt_str}");
    println!("OUTPUT: {text}");
    println!("wall:                {wall:.1}s for {generated} tokens");
    println!("tok/s:               {:.3}", generated as f64 / wall);
    println!("target forwards:     {target_forwards}");
    println!("draft forwards:      {draft_forwards}");
    println!("accepted drafts:     {accepted_total}");
    println!("accept rate:         {:.1}%",
        100.0 * accepted_total as f64 / (target_forwards * k_max).max(1) as f64);

    Ok(())
}
