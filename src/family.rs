//! Model family detection + config adaptation.
//!
//! For Llama-architecture-compatible families (Llama, Mistral, Qwen2,
//! InternLM2), we parse the family-specific HF config and project it
//! onto candle's `llama::Config` so the same streaming Block can run
//! all of them. Architecturally-different families (Mixtral MoE,
//! ChatGLM, Baichuan-13B ALiBi) are listed but unsupported here and
//! need their own streaming blocks.

use std::path::Path;

use candle_transformers::models::llama::{Config, LlamaConfig};
use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Llama,
    Mistral,
    Qwen2,
    InternLM2,
    // Architecturally distinct — not yet supported by StreamingLlama:
    Mixtral,
    ChatGlm,
    Baichuan,
}

impl Family {
    pub fn is_llama_compatible(self) -> bool {
        matches!(
            self,
            Family::Llama | Family::Mistral | Family::Qwen2 | Family::InternLM2
        )
    }
}

#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
}

pub fn detect(model_dir: &Path) -> Result<Family> {
    let probe: ModelTypeProbe =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    family_from_probe(&probe.model_type, &probe.architectures).ok_or_else(|| {
        Error::Msg(format!(
            "unknown model family: {} {:?}",
            probe.model_type, probe.architectures
        ))
    })
}

fn family_from_probe(model_type: &str, architectures: &[String]) -> Option<Family> {
    let mt = model_type.to_ascii_lowercase();
    if mt.contains("mixtral") {
        return Some(Family::Mixtral);
    }
    if mt.contains("mistral") {
        return Some(Family::Mistral);
    }
    if mt.contains("qwen2") {
        return Some(Family::Qwen2);
    }
    if mt.contains("internlm") {
        return Some(Family::InternLM2);
    }
    if mt.contains("chatglm") || mt.contains("glm") {
        return Some(Family::ChatGlm);
    }
    if mt.contains("baichuan") {
        return Some(Family::Baichuan);
    }
    if mt.contains("llama") {
        return Some(Family::Llama);
    }
    // Fall back to architectures[] for repos that omit model_type.
    for a in architectures {
        let al = a.to_ascii_lowercase();
        if al.contains("mixtral") {
            return Some(Family::Mixtral);
        }
        if al.contains("mistral") {
            return Some(Family::Mistral);
        }
        if al.contains("qwen2") {
            return Some(Family::Qwen2);
        }
        if al.contains("internlm") {
            return Some(Family::InternLM2);
        }
        if al.contains("chatglm") || al.contains("glm") {
            return Some(Family::ChatGlm);
        }
        if al.contains("baichuan") {
            return Some(Family::Baichuan);
        }
        if al.contains("llama") {
            return Some(Family::Llama);
        }
    }
    None
}

/// Build a candle `llama::Config` from the model's config.json, applying
/// family-specific adaptations. Returns `None` for families that need a
/// different Block implementation.
pub fn load_llama_compatible_config(model_dir: &Path) -> Result<Option<Config>> {
    let family = detect(model_dir)?;
    if !family.is_llama_compatible() {
        return Ok(None);
    }

    // All four Llama-compatible families have config.json shapes that
    // deserialize into candle's LlamaConfig (modulo extra fields it
    // ignores). Differences relative to vanilla Llama:
    //   - Mistral: `sliding_window` field present — ignored here; only
    //     matters when seq_len > sliding_window, which our short smoke
    //     tests don't hit. Real parity will need a windowed mask.
    //   - Qwen2: q/k/v projections have a learned bias. Our Block uses
    //     `linear_no_bias`, so Qwen2 weights will load but be off until
    //     a bias-aware attention variant is added.
    //   - InternLM2: same bias situation as Qwen2 for some variants.
    let llama_cfg: LlamaConfig =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    Ok(Some(llama_cfg.into_config(false)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_dispatch() {
        assert_eq!(family_from_probe("llama", &[]), Some(Family::Llama));
        assert_eq!(family_from_probe("mistral", &[]), Some(Family::Mistral));
        assert_eq!(family_from_probe("mixtral", &[]), Some(Family::Mixtral));
        assert_eq!(family_from_probe("qwen2", &[]), Some(Family::Qwen2));
        assert_eq!(
            family_from_probe("", &["MistralForCausalLM".into()]),
            Some(Family::Mistral)
        );
        assert_eq!(family_from_probe("", &[]), None);
    }

    #[test]
    fn llama_compatibility_flags() {
        assert!(Family::Llama.is_llama_compatible());
        assert!(Family::Mistral.is_llama_compatible());
        assert!(!Family::Mixtral.is_llama_compatible());
        assert!(!Family::ChatGlm.is_llama_compatible());
    }
}
