//! Layer-wise / memory-efficient Llama inference via Candle.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::shard::{ensure_sharded, shard_weight_files, ShardConfig, ShardPaths};

/// High-level Llama loader with optional per-layer sharding on disk.
pub struct LayeredLlama {
    model: Llama,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
    max_seq_len: usize,
}

pub struct LoadOptions {
    pub device: Device,
    pub dtype: DType,
    pub max_seq_len: usize,
    pub shard_config: ShardConfig,
    pub layer_shards_dir: Option<PathBuf>,
    pub hf_token: Option<String>,
    /// Enable async H2D streaming (CUDA only). When true and the
    /// device is CUDA, `StreamingLlama` uses
    /// [`crate::async_upload::AsyncWeightLoader`] to overlap layer
    /// weight transfer with current-layer compute. Falls back
    /// silently to the synchronous path on non-CUDA devices.
    pub async_h2d: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: best_available_device(),
            dtype: DType::F16,
            max_seq_len: 512,
            shard_config: ShardConfig::default(),
            layer_shards_dir: None,
            hf_token: None,
            async_h2d: false,
        }
    }
}

/// Pick the best GPU we can. Order: Metal (if compiled in and on macOS)
/// → CUDA (if compiled in) → CPU. Lets a single binary work across
/// macOS and Linux without code changes at call sites.
pub fn best_available_device() -> Device {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::cuda_if_available(0) {
            return d;
        }
    }
    Device::Cpu
}

impl LayeredLlama {
    /// Load from a local HuggingFace model directory or hub repo id.
    pub fn from_pretrained(model_id: &str, opts: LoadOptions) -> Result<Self> {
        let (model_dir, is_hub) = resolve_model_dir(model_id, opts.hf_token.as_deref())?;

        let config_path = model_dir.join("config.json");
        let llama_cfg: LlamaConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        let config = llama_cfg.into_config(false);

        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(Error::Tokenizer)?;

        let shard_paths = if is_hub || model_dir.join("model.safetensors.index.json").exists() {
            Some(ensure_sharded(
                &model_dir,
                opts.layer_shards_dir.as_deref(),
                &opts.shard_config,
            )?)
        } else {
            None
        };

        let weight_files = weight_files_for_model(&model_dir, shard_paths.as_ref())?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, opts.dtype, &opts.device)?
        };
        let model = Llama::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device: opts.device,
            config,
            max_seq_len: opts.max_seq_len,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Greedy / sampled text generation.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<String> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(Error::Tokenizer)?;
        let mut tokens: Vec<u32> = tokens.get_ids().to_vec();
        if tokens.len() > self.max_seq_len {
            tokens.truncate(self.max_seq_len);
        }

        let mut logits_processor =
            LogitsProcessor::new(299_792_458, Some(temperature).filter(|&t| t > 0.0), top_p);

        let mut cache = Cache::new(true, DType::F32, &self.config, &self.device)?;

        let mut index_pos = 0usize;
        for _ in 0..max_new_tokens {
            let context_size = if cache.use_kv_cache && index_pos > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = tokens.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&tokens[ctxt..], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_ids, index_pos, &mut cache)?;
            let logits = logits.i(0)?;
            index_pos += context_size;

            let next = logits_processor.sample(&logits)?;
            tokens.push(next);

            if let Some(eos) = self.config.eos_token_id.as_ref() {
                let eos_ids: Vec<u32> = match eos {
                    candle_transformers::models::llama::LlamaEosToks::Single(id) => vec![*id],
                    candle_transformers::models::llama::LlamaEosToks::Multiple(ids) => ids.clone(),
                };
                if eos_ids.contains(&next) {
                    break;
                }
            }
        }

        self.tokenizer
            .decode(&tokens, true)
            .map_err(Error::Tokenizer)
    }
}

pub(crate) fn resolve_model_dir(model_id: &str, hf_token: Option<&str>) -> Result<(PathBuf, bool)> {
    resolve_model_dir_inner(model_id, hf_token, /* download_weights = */ true)
}

/// Like `resolve_model_dir` but skips the (potentially huge) weight
/// download. Use when per-layer shards already exist on disk and only
/// config/tokenizer/index files are needed from the hub.
pub(crate) fn resolve_model_dir_metadata_only(
    model_id: &str,
    hf_token: Option<&str>,
) -> Result<(PathBuf, bool)> {
    resolve_model_dir_inner(model_id, hf_token, /* download_weights = */ false)
}

fn resolve_model_dir_inner(
    model_id: &str,
    _hf_token: Option<&str>,
    download_weights: bool,
) -> Result<(PathBuf, bool)> {
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Ok((path, false));
    }

    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    let config = repo.get("config.json")?;
    let model_dir = config
        .parent()
        .ok_or_else(|| Error::Msg("invalid config path from hub".into()))?
        .to_path_buf();

    // tokenizer.json is small but always needed by callers.
    let _ = repo.get("tokenizer.json")?;

    // The weight index is small and useful for layer-prefix discovery
    // even when we won't download the weights themselves.
    let index_path = repo.get("model.safetensors.index.json");

    if download_weights {
        match index_path {
            Ok(p) => {
                let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&p)?)?;
                if let Some(map) = index["weight_map"].as_object() {
                    let mut files: Vec<String> = map
                        .values()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect();
                    files.sort();
                    files.dedup();
                    for f in files {
                        let _ = repo.get(&f)?;
                    }
                } else {
                    let _ = repo.get("model.safetensors")?;
                }
            }
            Err(_) => {
                let _ = repo.get("model.safetensors")?;
            }
        }
    }

    Ok((model_dir, true))
}

fn weight_files_for_model(model_dir: &Path, shards: Option<&ShardPaths>) -> Result<Vec<PathBuf>> {
    if let Some(sp) = shards {
        return shard_weight_files(&sp.root, &sp.layers);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        let mut files: Vec<PathBuf> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .filter_map(|v| v.as_str())
            .map(|f| model_dir.join(f))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files);
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    Err(Error::Msg(format!(
        "no weights found under {}",
        model_dir.display()
    )))
}
