//! Per-layer GPU residency Llama inference — the core streaming primitive.
//!
//! Embedding, final norm, and lm_head stay resident on the device. Each
//! transformer block is built from its on-disk shard via mmap on every
//! forward call, runs, and is dropped before the next block loads. This
//! lets very large models (70B+) fit on a single small GPU at the cost
//! of repeated disk → device copies.
//!
//! Block/attention/MLP/Cache logic is adapted from
//! `candle-transformers::models::llama` (Apache-2.0); candle's versions
//! are private to its crate, so we inline a streaming-friendly copy here.
//! See https://github.com/huggingface/candle for the upstream.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias as linear, Embedding, Linear, Module, RmsNorm, VarBuilder,
};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType, LlamaEosToks};
use candle_transformers::utils::{build_causal_mask, repeat_kv};
use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::family;
use crate::inference::{resolve_model_dir, resolve_model_dir_metadata_only, LoadOptions};
use crate::shard::{ensure_sharded, layered_shards_complete, ShardPaths};

// Projection dispatch: at construction time, inspect the weight's device.
// CPU weights → ParallelLinear (rayon-parallel, ~4x candle's single-thread
// CPU matmul at our shapes). Any other device → candle Linear. This lets a
// single Block type serve all three residency tiers (GPU resident, CPU
// resident, mmap-streamed) and pick the fastest matmul for each.
#[cfg(feature = "parallel-cpu")]
pub enum Proj {
    Parallel(crate::parallel_linear::ParallelLinear),
    Standard(Linear),
}

#[cfg(feature = "parallel-cpu")]
impl Proj {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Parallel(p) => p
                .forward(x)
                .map_err(|e| candle_core::Error::Msg(format!("ParallelLinear: {e}"))),
            Self::Standard(l) => l.forward(x),
        }
    }
}

#[cfg(not(feature = "parallel-cpu"))]
use candle_nn::Linear as Proj;

#[cfg(feature = "parallel-cpu")]
fn build_proj(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Proj> {
    let l = linear(in_features, out_features, vb)?;
    if l.weight().device().is_cpu() {
        Ok(Proj::Parallel(
            crate::parallel_linear::ParallelLinear::new(l.weight().clone(), None)?,
        ))
    } else {
        Ok(Proj::Standard(l))
    }
}

#[cfg(not(feature = "parallel-cpu"))]
fn build_proj(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Proj> {
    Ok(linear(in_features, out_features, vb)?)
}

pub struct StreamCache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl StreamCache {
    pub fn new(use_kv_cache: bool, dtype: DType, cfg: &Config, device: &Device) -> Result<Self> {
        let theta = match &cfg.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => default_inv_freq(cfg),
            Some(s) => {
                let low = s.original_max_position_embeddings as f32 / s.low_freq_factor;
                let high = s.original_max_position_embeddings as f32 / s.high_freq_factor;
                default_inv_freq(cfg)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2.0 * PI / freq;
                        if wavelen < high {
                            freq
                        } else if wavelen > low {
                            freq / s.factor
                        } else {
                            let smooth = (s.original_max_position_embeddings as f32 / wavelen
                                - s.low_freq_factor)
                                / (s.high_freq_factor - s.low_freq_factor);
                            (1.0 - smooth) * freq / s.factor + smooth * freq
                        }
                    })
                    .collect()
            }
        };
        let theta = Tensor::new(theta, device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; cfg.num_hidden_layers],
            cos,
            sin,
            device: device.clone(),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(m) = self.masks.get(&(seq_len, kv_len)) {
            return Ok(m.clone());
        }
        let m = build_causal_mask(seq_len, index_pos, &self.device)?;
        self.masks.insert((seq_len, kv_len), m.clone());
        Ok(m)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    Ok(mask.where_cond(&on_true, on_false)?)
}

struct Attn {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

impl Attn {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        Ok(Self {
            q_proj: build_proj(size_in, size_q, vb.pp("q_proj"))?,
            k_proj: build_proj(size_in, size_kv, vb.pp("k_proj"))?,
            v_proj: build_proj(size_in, size_kv, vb.pp("v_proj"))?,
            o_proj: build_proj(size_q, size_in, vb.pp("o_proj"))?,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }

    fn apply_rope(&self, x: &Tensor, index_pos: usize, cache: &StreamCache) -> Result<Tensor> {
        let (_b, _, seq_len, _h) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        Ok(candle_nn::rotary_emb::rope(x, &cos, &sin)?)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut StreamCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rope(&q, index_pos, cache)?;
        let mut k = self.apply_rope(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((ck, cv)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[ck, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cv, &v], 2)?.contiguous()?;
                let k_len = k.dims()[1];
                if k_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_len = v.dims()[1];
                if v_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        Ok(self.o_proj.forward(&y)?)
    }
}

struct Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate: build_proj(h, i, vb.pp("gate_proj"))?,
            up: build_proj(h, i, vb.pp("up_proj"))?,
            down: build_proj(i, h, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        Ok(self.down.forward(&y)?)
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: Attn,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn: Attn::load(vb.pp("self_attn"), cfg)?,
            rms_2,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut StreamCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.rms_1.forward(x)?;
        let h = (self.attn.forward(&h, index_pos, block_idx, cache)? + residual)?;
        let residual = &h;
        let h = (self.mlp.forward(&self.rms_2.forward(&h)?)? + residual)?;
        Ok(h)
    }
}

/// Per-layer streaming Llama. Only embed + final norm + lm_head are
/// resident on the device; each transformer block's weights are loaded
/// from disk into device memory per forward call and freed afterward.
pub struct StreamingLlama {
    wte: Embedding,
    ln_f: RmsNorm,
    lm_head: Linear,
    layer_files: Vec<PathBuf>,
    cfg: Config,
    device: Device,
    dtype: DType,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    /// When present, the forward loop uses async H2D + double-buffer
    /// instead of the synchronous mmap path. CUDA-only.
    #[cfg(feature = "cuda")]
    async_loader: Option<std::sync::Mutex<crate::async_upload::AsyncWeightLoader>>,
    /// GPU-resident F16 transformer blocks (one per layer). Layers in
    /// `[0, resident_blocks.len())` stay on the device across forward
    /// calls — eliminates per-token streaming + Block::load cost.
    /// Each Llama-2-70B F16 layer is ~1.75 GB on the device, so the
    /// safe ceiling on a 12 GiB GPU is ~6 layers alongside KV cache
    /// and activations.
    resident_blocks: Vec<Block>,
    /// CPU-resident F16 transformer blocks. Layers in
    /// `[resident_blocks.len(), resident_blocks.len() + cpu_resident_blocks.len())`
    /// run on host with hidden state transferred GPU↔CPU at the
    /// boundary. Trades two small transfers per token for eliminating
    /// the ~500 ms/layer streaming overhead. Llama-2-70B F16 layer is
    /// ~1.75 GB host RAM; ~70 layers fits a 125 GB box (tight).
    cpu_resident_blocks: Vec<Block>,
    /// Persistent KV cache for the CPU-resident layers. Must survive
    /// across forward() calls or the model drifts immediately.
    cpu_cache: std::sync::Mutex<Option<StreamCache>>,
    /// Optional Q4/Q8 GGUF shards for the streamed tier — when set,
    /// the streamed layers load from these GGUF files via QMatMul
    /// instead of from the F16 safetensors. Lets the residency tiers
    /// run at F16 quality while the PCIe-bound streamed tier runs at
    /// ~3x less bandwidth. Indexed parallel to `layer_files`.
    quantized_stream_files: Option<Vec<PathBuf>>,
    /// Persistent KV cache for the quantized streamed tier. Same
    /// lifetime concern as `cpu_cache`.
    quantized_stream_cache: std::sync::Mutex<Option<crate::streaming_quantized::QuantStreamCache>>,
}

impl StreamingLlama {
    /// Build from resident weights + per-layer shard paths.
    ///
    /// `layer_files[i]` must be a safetensors file whose keys are the
    /// fully-qualified names for `model.layers.i.*` (matching the on-disk
    /// layout produced by [`crate::shard::shard_model`]).
    pub fn new(
        wte_vb: VarBuilder,
        norm_vb: VarBuilder,
        lm_head_vb: VarBuilder,
        layer_files: Vec<PathBuf>,
        cfg: Config,
        device: Device,
        dtype: DType,
        tokenizer: Tokenizer,
        max_seq_len: usize,
    ) -> Result<Self> {
        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            wte_vb.pp("model.embed_tokens"),
        )?;
        let ln_f =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, norm_vb.pp("model.norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, lm_head_vb.pp("lm_head"))?
        };
        Ok(Self {
            wte,
            ln_f,
            lm_head,
            layer_files,
            cfg,
            device,
            dtype,
            tokenizer,
            max_seq_len,
            #[cfg(feature = "cuda")]
            async_loader: None,
            resident_blocks: Vec::new(),
            cpu_resident_blocks: Vec::new(),
            cpu_cache: std::sync::Mutex::new(None),
            quantized_stream_files: None,
            quantized_stream_cache: std::sync::Mutex::new(None),
        })
    }

    /// Point the streamed tier at a directory of Q4/Q8 GGUF shards
    /// (`model.layers.N.gguf`). When set, tier 3 in `forward()` loads
    /// each streamed layer from GGUF via `QMatMul` rather than from
    /// F16 safetensors. Resident tiers (GPU + CPU) keep their F16
    /// weights. The mix lets 80% of weights stay F16 (full quality on
    /// the per-token-touched layers) while the 20% that bottleneck on
    /// PCIe drop to Q4 — net ~2x on F16 hybrid wall-time at near-F16
    /// quality, since the dequantization happens after the H2D so the
    /// per-token GPU activation path is unchanged.
    pub fn enable_quantized_streaming<P: AsRef<std::path::Path>>(
        &mut self,
        gguf_dir: P,
    ) -> Result<()> {
        let dir = gguf_dir.as_ref();
        let files = (0..self.layer_files.len())
            .map(|i| dir.join(format!("model.layers.{i}.gguf")))
            .collect::<Vec<_>>();
        for p in &files {
            if !p.exists() {
                return Err(Error::Msg(format!(
                    "enable_quantized_streaming: missing shard {}",
                    p.display()
                )));
            }
        }
        self.quantized_stream_files = Some(files);
        Ok(())
    }

    /// Pre-load `n` layers into device memory and keep them resident
    /// across forward calls. Cheapest for layers we know we'll always
    /// hit; bounded by GPU memory (~6 F16 layers on a 12 GiB card).
    pub fn pin_resident_layers(&mut self, n: usize) -> Result<()> {
        let n = n.min(self.layer_files.len());
        self.resident_blocks.clear();
        self.resident_blocks.reserve(n);
        for i in 0..n {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.layer_files[i]),
                    self.dtype,
                    &self.device,
                )?
            };
            let block = Block::load(vb.pp(format!("model.layers.{i}")), &self.cfg)?;
            self.resident_blocks.push(block);
        }
        Ok(())
    }

    /// Pre-load `n` layers (starting after the GPU-resident set) into
    /// host RAM as CPU-device blocks. They run after the GPU-resident
    /// layers and before any streamed layers; hidden state transfers
    /// GPU→CPU then CPU→GPU at the boundary.
    pub fn pin_cpu_resident_layers(&mut self, n: usize) -> Result<()> {
        let gpu_count = self.resident_blocks.len();
        let max = self.layer_files.len().saturating_sub(gpu_count);
        let n = n.min(max);
        self.cpu_resident_blocks.clear();
        self.cpu_resident_blocks.reserve(n);
        for offset in 0..n {
            let i = gpu_count + offset;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.layer_files[i]),
                    self.dtype,
                    &Device::Cpu,
                )?
            };
            let block = Block::load(vb.pp(format!("model.layers.{i}")), &self.cfg)?;
            self.cpu_resident_blocks.push(block);
        }
        Ok(())
    }

    /// Load from a HuggingFace repo id or local model directory.
    ///
    /// Mirrors [`crate::inference::LayeredLlama::from_pretrained`] but
    /// wires up the streaming forward path. Embed/norm/lm_head shards
    /// each get their own VarBuilder so they stay resident; transformer
    /// blocks are loaded per-layer at forward time.
    pub fn from_pretrained(model_id: &str, opts: LoadOptions) -> Result<Self> {
        // Fast path: if per-layer shards already exist under the configured
        // layer_shards_dir, skip the (possibly 100GB+) raw-weights download.
        let shards_complete = opts
            .layer_shards_dir
            .as_deref()
            .map(|d| layered_shards_complete(&d.join(&opts.shard_config.splitted_dir_name)))
            .unwrap_or(false);

        let (model_dir, _) = if shards_complete {
            resolve_model_dir_metadata_only(model_id, opts.hf_token.as_deref())?
        } else {
            resolve_model_dir(model_id, opts.hf_token.as_deref())?
        };
        let cfg = family::load_llama_compatible_config(&model_dir)?.ok_or_else(|| {
            Error::Msg(format!(
                "model at {} uses a non-Llama-compatible architecture; \
                 streaming for this family is not implemented yet",
                model_dir.display()
            ))
        })?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(Error::Tokenizer)?;

        let ShardPaths { root, layers } = ensure_sharded(
            &model_dir,
            opts.layer_shards_dir.as_deref(),
            &opts.shard_config,
        )?;

        let shard_path = |name: &str| root.join(format!("{name}.safetensors"));
        let embed_path = shard_path(&opts.shard_config.layer_names.embed);
        let norm_path = shard_path(&opts.shard_config.layer_names.norm);
        let lm_head_path = shard_path(&opts.shard_config.layer_names.lm_head);

        let wte_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[embed_path], opts.dtype, &opts.device)?
        };
        let norm_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[norm_path], opts.dtype, &opts.device)? };
        let lm_head_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[lm_head_path], opts.dtype, &opts.device)?
        };

        let layer_prefix = &opts.shard_config.layer_names.layer_prefix;
        let layer_files: Vec<PathBuf> = layers
            .iter()
            .filter(|l| l.starts_with(&format!("{layer_prefix}.")))
            .map(|l| shard_path(l))
            .collect();

        let mut s = Self::new(
            wte_vb,
            norm_vb,
            lm_head_vb,
            layer_files,
            cfg,
            opts.device.clone(),
            opts.dtype,
            tokenizer,
            opts.max_seq_len,
        )?;
        // If async H2D was requested AND we have a CUDA device AND at
        // least one layer file, build the AsyncWeightLoader using the
        // first layer's header as the schema sample. On non-CUDA or
        // when async_h2d=false this stays None and forward() uses the
        // synchronous mmap path.
        #[cfg(feature = "cuda")]
        if opts.async_h2d {
            if matches!(opts.device, Device::Cuda(_)) && !s.layer_files.is_empty() {
                let loader = crate::async_upload::AsyncWeightLoader::new(
                    &opts.device,
                    &s.layer_files[0],
                    opts.dtype,
                )?;
                s.async_loader = Some(std::sync::Mutex::new(loader));
                tracing::info!("StreamingLlama: async H2D loader enabled");
            } else {
                tracing::warn!(
                    "StreamingLlama: async_h2d requested but device is not CUDA — falling back to sync path"
                );
            }
        }
        // GPU + CPU residency knobs via env, matching the Q4 path so
        // the bench scripts pick them up uniformly.
        if let Ok(s_val) = std::env::var("RUSTYLLM_RESIDENT_LAYERS") {
            if let Ok(n) = s_val.parse::<usize>() {
                s.pin_resident_layers(n)?;
            }
        }
        if let Ok(s_val) = std::env::var("RUSTYLLM_CPU_RESIDENT_LAYERS") {
            if let Ok(n) = s_val.parse::<usize>() {
                s.pin_cpu_resident_layers(n)?;
            }
        }
        Ok(s)
    }

    /// Greedy / sampled text generation, mirroring `LayeredLlama::generate`.
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

        let mut lp =
            LogitsProcessor::new(299_792_458, Some(temperature).filter(|&t| t > 0.0), top_p);
        let mut cache = StreamCache::new(true, self.dtype, &self.cfg, &self.device)?;

        let mut index_pos = 0usize;
        for _ in 0..max_new_tokens {
            let context_size = if cache.use_kv_cache && index_pos > 0 {
                1
            } else {
                tokens.len()
            };
            let ctxt = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[ctxt..], &self.device)?.unsqueeze(0)?;
            let logits = self.forward(&input, index_pos, &mut cache)?;
            let logits = logits.i(0)?;
            index_pos += context_size;

            let next = lp.sample(&logits)?;
            tokens.push(next);

            if let Some(eos) = self.cfg.eos_token_id.as_ref() {
                let stop = match eos {
                    LlamaEosToks::Single(id) => next == *id,
                    LlamaEosToks::Multiple(ids) => ids.contains(&next),
                };
                if stop {
                    break;
                }
            }
        }

        self.tokenizer
            .decode(&tokens, true)
            .map_err(Error::Tokenizer)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut StreamCache,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut x = self.wte.forward(input_ids)?;

        // Tier 1: GPU-resident layers. Always on the device.
        for (i, block) in self.resident_blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, i, cache)?;
        }

        // Tier 2: CPU-resident layers. Transfer hidden state GPU→CPU,
        // run, transfer back. KV cache for this stretch lives on a
        // persistent CPU cache so history accumulates across tokens.
        let cpu_start = self.resident_blocks.len();
        if !self.cpu_resident_blocks.is_empty() {
            let cpu_dev = Device::Cpu;
            let mut x_cpu = x.to_device(&cpu_dev)?;
            let mut cpu_cache_guard = self
                .cpu_cache
                .lock()
                .map_err(|e| Error::Msg(format!("cpu_cache mutex poisoned: {e}")))?;
            if cpu_cache_guard.is_none() {
                *cpu_cache_guard =
                    Some(StreamCache::new(cache.use_kv_cache, self.dtype, &self.cfg, &cpu_dev)?);
            }
            let cpu_cache = cpu_cache_guard.as_mut().unwrap();
            for (offset, block) in self.cpu_resident_blocks.iter().enumerate() {
                let i = cpu_start + offset;
                x_cpu = block.forward(&x_cpu, index_pos, i, cpu_cache)?;
            }
            x = x_cpu.to_device(&self.device)?;
        }

        // Tier 3: streamed layers (everything not covered above).
        let stream_start = self.resident_blocks.len() + self.cpu_resident_blocks.len();

        // Tier 3a: quantized GGUF streaming (mixed precision). When set
        // we read each streamed layer from a Q4_K GGUF instead of the
        // F16 safetensors — ~3x less PCIe traffic per layer. QMatMul
        // dequantises on device into the existing F32 activation path.
        if let Some(qfiles) = self.quantized_stream_files.as_ref() {
            if stream_start < qfiles.len() {
                let mut qcache_guard = self
                    .quantized_stream_cache
                    .lock()
                    .map_err(|e| Error::Msg(format!("quantized_stream_cache mutex: {e}")))?;
                if qcache_guard.is_none() {
                    *qcache_guard =
                        Some(crate::streaming_quantized::QuantStreamCache::new(
                            cache.use_kv_cache,
                            DType::F32,
                            &self.cfg,
                            &self.device,
                        )?);
                }
                let qcache = qcache_guard.as_mut().unwrap();
                // QMatMul wants F32 activations.
                let mut xq = x.to_dtype(DType::F32)?;
                for offset in 0..(qfiles.len() - stream_start) {
                    let i = stream_start + offset;
                    let mut layer = crate::streaming_quantized::GgufLayer::open(
                        &qfiles[i],
                        &self.device,
                    )?;
                    let qblock = crate::streaming_quantized::QBlock::load(
                        &mut layer, i, &self.cfg,
                    )?;
                    xq = qblock.forward(&xq, index_pos, i, qcache)?;
                }
                x = xq.to_dtype(self.dtype)?;
                let x = self.ln_f.forward(&x)?;
                let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
                let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?;
                return Ok(logits);
            }
        }

        #[cfg(feature = "cuda")]
        if let Some(loader_lock) = self.async_loader.as_ref() {
            if stream_start < self.layer_files.len() {
                let mut loader = loader_lock
                    .lock()
                    .map_err(|e| Error::Msg(format!("loader mutex poisoned: {e}")))?;
                loader.prefetch(&self.layer_files[stream_start])?;
                for offset in 0..(self.layer_files.len() - stream_start) {
                    let i = stream_start + offset;
                    let vb = loader.take_var_builder(self.dtype, i)?;
                    if i + 1 < self.layer_files.len() {
                        loader.prefetch(&self.layer_files[i + 1])?;
                    }
                    let block = Block::load(vb.pp(format!("model.layers.{i}")), &self.cfg)?;
                    x = block.forward(&x, index_pos, i, cache)?;
                }
            }
        } else {
            for offset in 0..(self.layer_files.len() - stream_start) {
                let i = stream_start + offset;
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&self.layer_files[i]),
                        self.dtype,
                        &self.device,
                    )?
                };
                let block = Block::load(vb.pp(format!("model.layers.{i}")), &self.cfg)?;
                x = block.forward(&x, index_pos, i, cache)?;
            }
        }

        #[cfg(not(feature = "cuda"))]
        for offset in 0..(self.layer_files.len() - stream_start) {
            let i = stream_start + offset;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(&self.layer_files[i]),
                    self.dtype,
                    &self.device,
                )?
            };
            let block = Block::load(vb.pp(format!("model.layers.{i}")), &self.cfg)?;
            x = block.forward(&x, index_pos, i, cache)?;
        }

        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?;
        Ok(logits)
    }
}
