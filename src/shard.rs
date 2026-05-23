//! Split HuggingFace checkpoints into per-layer safetensor shards.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::error::{Error, Result};

/// Standard Llama-style layer name prefixes.
#[derive(Debug, Clone)]
pub struct LayerNames {
    pub embed: String,
    pub layer_prefix: String,
    pub norm: String,
    pub lm_head: String,
}

impl Default for LayerNames {
    fn default() -> Self {
        Self {
            embed: "model.embed_tokens".into(),
            layer_prefix: "model.layers".into(),
            norm: "model.norm".into(),
            lm_head: "lm_head".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Family;

    #[test]
    fn for_family_picks_correct_layer_prefix() {
        let llama = ShardConfig::for_family(Family::Llama);
        assert_eq!(llama.layer_names.layer_prefix, "model.layers");

        let chatglm = ShardConfig::for_family(Family::ChatGlm);
        assert_eq!(
            chatglm.layer_names.layer_prefix,
            "transformer.encoder.layers"
        );

        let mixtral = ShardConfig::for_family(Family::Mixtral);
        assert_eq!(mixtral.layer_names.layer_prefix, "model.layers");
    }
}

#[derive(Debug, Clone)]
pub struct ShardConfig {
    pub layer_names: LayerNames,
    pub splitted_dir_name: String,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            layer_names: LayerNames::default(),
            splitted_dir_name: "splitted_model".into(),
        }
    }
}

impl ShardConfig {
    /// Return a ShardConfig with layer-name prefixes appropriate for
    /// the given model family. Llama/Mistral/Qwen2/InternLM2/Mixtral/
    /// Baichuan all use the default `model.layers.N` layout; ChatGLM is
    /// the outlier with `transformer.encoder.layers.N`. Using this
    /// avoids having to hand-construct `LayerNames` per family.
    pub fn for_family(family: crate::family::Family) -> Self {
        use crate::family::Family::*;
        let layer_names = match family {
            ChatGlm => crate::streaming_chatglm::default_layer_names(),
            Llama | Mistral | Qwen2 | InternLM2 | Mixtral | Baichuan => LayerNames::default(),
        };
        Self {
            layer_names,
            splitted_dir_name: "splitted_model".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardPaths {
    pub root: PathBuf,
    pub layers: Vec<String>,
}

/// Return path to layer shards, creating them from the HF checkpoint when missing.
pub fn ensure_sharded(
    checkpoint_dir: &Path,
    output_dir: Option<&Path>,
    config: &ShardConfig,
) -> Result<ShardPaths> {
    let layers = layer_prefixes(&config.layer_names, checkpoint_dir)?;
    let saving = output_dir
        .map(|p| p.join(&config.splitted_dir_name))
        .unwrap_or_else(|| checkpoint_dir.join(&config.splitted_dir_name));

    if all_layers_exist(&saving, &layers) {
        return Ok(ShardPaths {
            root: saving,
            layers,
        });
    }

    shard_model(checkpoint_dir, output_dir, config)
}

pub fn shard_model(
    checkpoint_dir: &Path,
    output_dir: Option<&Path>,
    config: &ShardConfig,
) -> Result<ShardPaths> {
    let layers = layer_prefixes(&config.layer_names, checkpoint_dir)?;
    let saving = output_dir
        .map(|p| p.join(&config.splitted_dir_name))
        .unwrap_or_else(|| checkpoint_dir.join(&config.splitted_dir_name));

    fs::create_dir_all(&saving)?;

    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(Error::Msg(format!("missing {}", index_path.display())));
    }

    let index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Msg("invalid weight_map".into()))?;

    let mut shard_files: HashMap<String, PathBuf> = HashMap::new();
    for file in weight_map.values().filter_map(|v| v.as_str()) {
        shard_files
            .entry(file.to_string())
            .or_insert_with(|| checkpoint_dir.join(file));
    }

    for layer in &layers {
        let layer_dot = format!("{layer}.");
        let tensor_names: Vec<&String> = weight_map
            .keys()
            .filter(|k| k.starts_with(&layer_dot) || *k == layer)
            .collect();

        if tensor_names.is_empty() {
            tracing::warn!(layer, "no tensors matched, skipping");
            continue;
        }

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut loaded_files: HashSet<String> = HashSet::new();

        for name in &tensor_names {
            let shard_file = weight_map
                .get(*name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Msg(format!("no shard for {name}")))?;
            if loaded_files.insert(shard_file.to_string()) {
                let path = shard_files
                    .get(shard_file)
                    .ok_or_else(|| Error::Msg(format!("missing file {shard_file}")))?;
                let loaded = candle_core::safetensors::load(path, &Device::Cpu)?;
                for (k, t) in loaded {
                    if tensor_names.iter().any(|n| **n == k) {
                        tensors.insert(k, t);
                    }
                }
            }
        }

        if tensors.is_empty() {
            return Err(Error::Msg(format!("empty shard for {layer}")));
        }

        let out = saving.join(format!("{layer}.safetensors"));
        candle_core::safetensors::save(&tensors, &out)?;
        fs::File::create(saving.join(format!("{layer}.safetensors.done")))?;
        tracing::info!(layer, path = %out.display(), count = tensors.len(), "wrote shard");
    }

    Ok(ShardPaths {
        root: saving,
        layers,
    })
}

fn layer_prefixes(names: &LayerNames, checkpoint_dir: &Path) -> Result<Vec<String>> {
    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Msg("invalid weight_map".into()))?;

    let prefix = format!("{}.", names.layer_prefix);
    let n_layers = weight_map
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .filter_map(|k| {
            k.strip_prefix(&prefix)?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut layers = vec![names.embed.clone()];
    for i in 0..n_layers {
        layers.push(format!("{}.{}", names.layer_prefix, i));
    }
    layers.push(names.norm.clone());
    layers.push(names.lm_head.clone());
    Ok(layers)
}

fn all_layers_exist(saving: &Path, layers: &[String]) -> bool {
    saving.exists()
        && layers.iter().all(|l| {
            saving.join(format!("{l}.safetensors")).exists()
                && saving.join(format!("{l}.safetensors.done")).exists()
        })
}

/// True if `splitted_dir` looks like a complete per-layer shard set,
/// without needing the original HuggingFace index.json. Detects by:
///   - directory exists
///   - has an embed shard + at least one model.layers.N shard
///   - every shard has a paired `.done` marker
///
/// Used to short-circuit weight downloads when the splits are already
/// present from a prior run.
pub fn layered_shards_complete(splitted_dir: &Path) -> bool {
    if !splitted_dir.is_dir() {
        return false;
    }
    let entries = match std::fs::read_dir(splitted_dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    let mut has_embed = false;
    let mut has_layer = false;
    let mut shards: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        let name = match p.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if let Some(stem) = name.strip_suffix(".safetensors") {
            shards.push(p.clone());
            if stem.contains("embed_tokens") {
                has_embed = true;
            } else if stem.contains(".layers.") {
                has_layer = true;
            }
        }
    }
    if !(has_embed && has_layer) {
        return false;
    }
    shards
        .iter()
        .all(|p| p.with_extension("safetensors.done").exists())
}

/// Collect safetensor paths for a layered model directory (for mmap loading).
pub fn shard_weight_files(shard_root: &Path, layers: &[String]) -> Result<Vec<PathBuf>> {
    layers
        .iter()
        .map(|l| {
            let p = shard_root.join(format!("{l}.safetensors"));
            if p.exists() {
                Ok(p)
            } else {
                Err(Error::MissingShard(p.display().to_string()))
            }
        })
        .collect()
}
