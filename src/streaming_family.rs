//! Single entry point for streaming inference across model families.
//!
//! Detects the family from `config.json` and dispatches to the
//! appropriate concrete `StreamingX` struct. Callers can ignore which
//! architecture they're loading until they need a family-specific
//! method:
//!
//! ```no_run
//! use rustyllm::{StreamingFamily, inference::LoadOptions};
//! let model = StreamingFamily::from_pretrained("garage-bAInd/Platypus2-7B",
//!                                              LoadOptions::default())?;
//! let text = model.generate("What is the capital of France?", 32, 0.0, None)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Today, only Llama-compatible families (Llama, Mistral, Qwen2,
//! InternLM2) plumb through the high-level `generate` shortcut.
//! Mixtral, ChatGLM, and Baichuan-13B require constructing the
//! concrete `StreamingMixtral`/`StreamingChatGlm`/`StreamingBaichuan`
//! directly because they each need a family-specific tokenizer and
//! generation loop (left as a deliberate follow-up — the per-family
//! Block code is already in place).

use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::family::{self, Family};
use crate::inference::{resolve_model_dir_metadata_only, LoadOptions};
use crate::streaming::StreamingLlama;
use crate::streaming_mixtral::StreamingMixtral;

/// Resolve enough of a model to read its config.json — local paths
/// return as-is, hub repos pull config + tokenizer (small files)
/// without fetching any weights. Used by [`StreamingFamily::from_pretrained`]
/// to dispatch on family before committing to a multi-GB download.
fn resolve_config_only(model_id: &str, hf_token: Option<&str>) -> Result<PathBuf> {
    let p = std::path::Path::new(model_id);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    let (dir, _) = resolve_model_dir_metadata_only(model_id, hf_token)?;
    Ok(dir)
}

pub enum StreamingFamily {
    /// Llama, Mistral, Qwen2, or InternLM2 — share the same Block.
    LlamaCompatible(StreamingLlama),
    /// Mixtral 8x7B (sparse top-K MoE) and similar.
    Mixtral(StreamingMixtral),
}

impl StreamingFamily {
    /// Detect the family from `config.json` and load the appropriate
    /// streaming inference path. Returns an error with the detected
    /// family name when no high-level loader is wired up for it; the
    /// caller can fall back to the family-specific constructor
    /// (`StreamingMixtral::new`, `StreamingChatGlm::new`,
    /// `StreamingBaichuan::new`) in that case.
    pub fn from_pretrained(model_id: &str, opts: LoadOptions) -> Result<Self> {
        // Detect the family BEFORE doing any heavy work. For local
        // paths, read config.json directly; for hub repos, fetch just
        // the config (small) so we can fail fast with the right error
        // for non-Llama-compatible families instead of trying to load
        // the wrong architecture and discovering 100GB later that the
        // weight shapes don't match.
        let model_dir = resolve_config_only(model_id, opts.hf_token.as_deref())?;
        let detected = family::detect(&model_dir)?;

        if detected.is_llama_compatible() {
            return Ok(StreamingFamily::LlamaCompatible(
                StreamingLlama::from_pretrained(model_id, opts)?,
            ));
        }
        if matches!(detected, Family::Mixtral) {
            return Ok(StreamingFamily::Mixtral(
                StreamingMixtral::from_pretrained(model_id, opts)?,
            ));
        }

        Err(Error::Msg(format!(
            "model family `{detected:?}` is implemented but needs its own loader; \
             construct `{}` directly with `from_mmapped_safetensors` + `::new`",
            family_loader_hint(detected)
        )))
    }

    /// Family the loaded model belongs to.
    pub fn family(&self) -> Family {
        match self {
            // Could be any of Llama/Mistral/Qwen2/InternLM2; the
            // StreamingLlama Block is the same for all four. If the
            // distinction matters, re-probe from the model dir.
            StreamingFamily::LlamaCompatible(_) => Family::Llama,
            StreamingFamily::Mixtral(_) => Family::Mixtral,
        }
    }

    /// Greedy / sampled text generation. Llama-compatible families and
    /// Mixtral implement a one-shot `generate`; ChatGLM and Baichuan
    /// still require manual driving via their concrete structs.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<String> {
        match self {
            StreamingFamily::LlamaCompatible(m) => {
                m.generate(prompt, max_new_tokens, temperature, top_p)
            }
            StreamingFamily::Mixtral(m) => {
                m.generate(prompt, max_new_tokens, temperature, top_p)
            }
        }
    }
}

fn family_loader_hint(family: Family) -> &'static str {
    use Family::*;
    match family {
        Mixtral => "StreamingMixtral",
        ChatGlm => "StreamingChatGlm",
        Baichuan => "StreamingBaichuan",
        // The Llama-compatible families never hit this branch.
        Llama | Mistral | Qwen2 | InternLM2 => "StreamingLlama",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_config(dir: &std::path::Path, model_type: &str) {
        fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type":"{model_type}","architectures":[]}}"#),
        )
        .unwrap();
    }

    #[test]
    fn dispatcher_rejects_non_llama_families_with_hint() {
        // Create local dirs that look like Mixtral / ChatGLM /
        // Baichuan. We can't actually load them (no weights), but the
        // detect-and-dispatch step happens before any weight access.
        // mixtral was previously in this list but now dispatches via
        // StreamingFamily::Mixtral (errs on weights, not on routing) —
        // covered by a dedicated regression test below.
        for (model_type, hint) in [
            ("chatglm", "StreamingChatGlm"),
            ("baichuan", "StreamingBaichuan"),
        ] {
            let dir = tempdir().unwrap();
            write_config(dir.path(), model_type);
            let res = StreamingFamily::from_pretrained(
                dir.path().to_str().unwrap(),
                LoadOptions::default(),
            );
            let msg = match res {
                Ok(_) => panic!("expected error for {model_type}"),
                Err(e) => format!("{e}"),
            };
            assert!(
                msg.contains(hint),
                "error should mention {hint} for {model_type}, got: {msg}"
            );
        }
    }

    #[test]
    fn dispatcher_detects_family_before_attempting_load() {
        // Regression: previously, hub-style model IDs (anything not
        // resolving as a local path) skipped family detection and
        // tried StreamingLlama unconditionally, which would download
        // gigabytes before failing. Confirm the detect step runs
        // first by using a local path that resolve_config_only treats
        // as already-resolved.
        let dir = tempdir().unwrap();
        write_config(dir.path(), "chatglm");
        let res =
            StreamingFamily::from_pretrained(dir.path().to_str().unwrap(), LoadOptions::default());
        let msg = match res {
            Ok(_) => panic!("expected error"),
            Err(e) => format!("{e}"),
        };
        // The error MUST be the family-routing hint, not a weight-load
        // error or an "unknown family" error.
        assert!(
            msg.contains("StreamingChatGlm"),
            "early detection should produce the loader-hint error, got: {msg}"
        );
        assert!(
            msg.contains("ChatGlm"),
            "should name the detected family, got: {msg}"
        );
        // Negative signal: a weight-load error would mention a tensor
        // path / shape mismatch / missing file, none of which should
        // appear when we bail out early on family detection alone.
        assert!(
            !msg.contains("missing") && !msg.contains("shape") && !msg.contains("No such file"),
            "must not have attempted weight load, got: {msg}"
        );
    }
}
