//! Per-layer streaming Mixtral (sparse Mixture-of-Experts) inference.
//!
//! Mixtral's decoder layer = Llama attention + an MoE block with N
//! experts and top-K routing. For Mixtral-8x7B: hidden=4096,
//! intermediate=14336, 8 experts, top-2. Each layer carries ~5.7GB of
//! F16 weights (vs ~0.4GB for vanilla Llama), almost entirely in the
//! experts. That fits on a 12GB GPU per layer, barely, with no room
//! for activations — so real streaming workloads should either (a)
//! enable 4/8-bit compression (task #4) or (b) stream experts within
//! a layer rather than whole layers. This module implements the
//! whole-layer flavor; sub-layer streaming is a follow-up.
//!
//! Block code is adapted from `candle-transformers::models::mixtral`
//! (Apache-2.0); candle's structs are private, so we inline a
//! streaming-friendly copy. See https://github.com/huggingface/candle.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias as linear, Activation, Embedding, Linear, RmsNorm, VarBuilder,
};
use candle_transformers::utils::repeat_kv;
use serde::Deserialize;

use crate::error::Result;
use crate::parallel_linear::ParallelLinear;

/// Mirror of streaming.rs::parallel_linear_no_bias — goes through
/// candle's `linear_no_bias` for VarBuilder name resolution then wraps
/// the weight in a rayon-parallel Linear for the actual fast forward.
fn parallel_linear_no_bias(
    in_features: usize,
    out_features: usize,
    vb: VarBuilder,
) -> Result<ParallelLinear> {
    let l = linear(in_features, out_features, vb)?;
    ParallelLinear::new(l.weight().clone(), None)
}

#[derive(Debug, Clone, Deserialize)]
pub struct MixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub num_experts_per_tok: usize,
    pub num_local_experts: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl MixtralConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

pub struct MixtralStreamCache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn inv_freq(cfg: &MixtralConfig) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect()
}

impl MixtralStreamCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        cfg: &MixtralConfig,
        device: &Device,
    ) -> Result<Self> {
        let theta = Tensor::new(inv_freq(cfg), device)?;
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
        let m = candle_transformers::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
        self.masks.insert((seq_len, kv_len), m.clone());
        Ok(m)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    Ok(mask.where_cond(&on_true, on_false)?)
}

struct Attention {
    q_proj: ParallelLinear,
    k_proj: ParallelLinear,
    v_proj: ParallelLinear,
    o_proj: ParallelLinear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &MixtralConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_sz = cfg.head_dim() * cfg.num_attention_heads;
        let kv_sz = cfg.head_dim() * cfg.num_key_value_heads;
        Ok(Self {
            q_proj: parallel_linear_no_bias(h, q_sz, vb.pp("q_proj"))?,
            k_proj: parallel_linear_no_bias(h, kv_sz, vb.pp("k_proj"))?,
            v_proj: parallel_linear_no_bias(h, kv_sz, vb.pp("v_proj"))?,
            o_proj: parallel_linear_no_bias(q_sz, h, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            max_pos: cfg.max_position_embeddings,
        })
    }

    fn apply_rope(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &MixtralStreamCache,
    ) -> Result<Tensor> {
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
        cache: &mut MixtralStreamCache,
    ) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rope(&q, index_pos, cache)?;
        let mut k = self.apply_rope(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((ck, cv)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[ck, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cv, &v], 2)?.contiguous()?;
                let klen = k.dims()[1];
                if klen > self.max_pos {
                    k = k
                        .narrow(D::Minus1, klen - self.max_pos, self.max_pos)?
                        .contiguous()?;
                }
                let vlen = v.dims()[1];
                if vlen > 2 * self.max_pos {
                    v = v
                        .narrow(D::Minus1, vlen - self.max_pos, self.max_pos)?
                        .contiguous()?;
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

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
        let y = att.matmul(&v.contiguous()?)?.to_dtype(x.dtype())?;
        let y = y.transpose(1, 2)?.reshape(&[b, seq_len, hidden])?;
        Ok(self.o_proj.forward(&y)?)
    }
}

struct Expert {
    w1: ParallelLinear,
    w2: ParallelLinear,
    w3: ParallelLinear,
    act: Activation,
}

impl Expert {
    fn load(vb: VarBuilder, cfg: &MixtralConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            w1: parallel_linear_no_bias(h, i, vb.pp("w1"))?,
            w2: parallel_linear_no_bias(i, h, vb.pp("w2"))?,
            w3: parallel_linear_no_bias(h, i, vb.pp("w3"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ParallelLinear doesn't impl Module yet, so call .forward(...)
        // directly instead of .apply(...).
        let lhs = self.w1.forward(x)?.apply(&self.act)?;
        let rhs = self.w3.forward(x)?;
        self.w2.forward(&(lhs * rhs)?)
    }
}

struct SparseMoe {
    gate: ParallelLinear,
    experts: Vec<Expert>,
    top_k: usize,
}

impl SparseMoe {
    fn load(vb: VarBuilder, cfg: &MixtralConfig) -> Result<Self> {
        let gate = parallel_linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        let evb = vb.pp("experts");
        for i in 0..cfg.num_local_experts {
            experts.push(Expert::load(evb.pp(i), cfg)?);
        }
        Ok(Self {
            gate,
            experts,
            top_k: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        let xs2 = xs.reshape(((), hidden))?;
        let router_logits = self.gate.forward(&xs2)?;
        let routing = candle_nn::ops::softmax_last_dim(&router_logits)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        // Greedy top-k selection per token (matches HF Mixtral reference).
        let n_experts = self.experts.len();
        let mut bucket_rows = vec![Vec::<u32>::new(); n_experts];
        let mut bucket_weights = vec![Vec::<f32>::new(); n_experts];
        for (row_idx, rw) in routing.iter().enumerate() {
            let mut order: Vec<u32> = (0..rw.len() as u32).collect();
            order.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
            let mut sum = 0f32;
            for &e in order.iter().take(self.top_k) {
                sum += rw[e as usize];
                bucket_rows[e as usize].push(row_idx as u32);
            }
            for &e in order.iter().take(self.top_k) {
                bucket_weights[e as usize].push(rw[e as usize] / sum);
            }
        }

        let mut ys = xs2.zeros_like()?;
        for (i, expert) in self.experts.iter().enumerate() {
            if bucket_rows[i].is_empty() {
                continue;
            }
            let rows = Tensor::new(bucket_rows[i].as_slice(), xs2.device())?;
            let weights =
                Tensor::new(bucket_weights[i].as_slice(), xs2.device())?.reshape(((), 1))?;
            let inputs = xs2.index_select(&rows, 0)?.reshape(((), hidden))?;
            let outputs = expert.forward(&inputs)?.broadcast_mul(&weights)?;
            ys = ys.index_add(&rows, &outputs, 0)?;
        }

        Ok(ys.reshape((b, seq_len, hidden))?)
    }
}

struct Block {
    attn: Attention,
    moe: SparseMoe,
    rms_1: RmsNorm,
    rms_2: RmsNorm,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &MixtralConfig) -> Result<Self> {
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            attn: Attention::load(vb.pp("self_attn"), cfg)?,
            moe: SparseMoe::load(vb.pp("block_sparse_moe"), cfg)?,
            rms_1,
            rms_2,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut MixtralStreamCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.rms_1.forward(x)?;
        let h = (self.attn.forward(&h, index_pos, block_idx, cache)? + residual)?;
        let residual = &h;
        Ok((self.moe.forward(&self.rms_2.forward(&h)?)? + residual)?)
    }
}

pub struct StreamingMixtral {
    wte: Embedding,
    ln_f: RmsNorm,
    lm_head: Linear,
    layer_files: Vec<PathBuf>,
    cfg: MixtralConfig,
    device: Device,
    dtype: DType,
    tokenizer: Option<tokenizers::Tokenizer>,
    max_seq_len: usize,
}

impl StreamingMixtral {
    /// Build from resident weights + per-layer shard paths.
    pub fn new(
        wte_vb: VarBuilder,
        norm_vb: VarBuilder,
        lm_head_vb: VarBuilder,
        layer_files: Vec<PathBuf>,
        cfg: MixtralConfig,
        device: Device,
        dtype: DType,
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
            tokenizer: None,
            max_seq_len: 512,
        })
    }

    /// Load from a HuggingFace repo id or local model directory.
    /// Mirrors `StreamingLlama::from_pretrained` but parses the
    /// family-specific `MixtralConfig` (with `num_local_experts` and
    /// `num_experts_per_tok`) instead of the standard `LlamaConfig`.
    pub fn from_pretrained(
        model_id: &str,
        opts: crate::inference::LoadOptions,
    ) -> Result<Self> {
        let shards_complete = opts
            .layer_shards_dir
            .as_deref()
            .map(|d| {
                crate::shard::layered_shards_complete(
                    &d.join(&opts.shard_config.splitted_dir_name),
                )
            })
            .unwrap_or(false);
        let (model_dir, _) = if shards_complete {
            crate::inference::resolve_model_dir_metadata_only(model_id, opts.hf_token.as_deref())?
        } else {
            crate::inference::resolve_model_dir(model_id, opts.hf_token.as_deref())?
        };

        let cfg: MixtralConfig =
            serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(crate::error::Error::Tokenizer)?;

        let crate::shard::ShardPaths { root, layers } = crate::shard::ensure_sharded(
            &model_dir,
            opts.layer_shards_dir.as_deref(),
            &opts.shard_config,
        )?;
        let shard_path = |name: &str| root.join(format!("{name}.safetensors"));
        let embed_path = shard_path(&opts.shard_config.layer_names.embed);
        let norm_path = shard_path(&opts.shard_config.layer_names.norm);
        let lm_head_path = shard_path(&opts.shard_config.layer_names.lm_head);

        let wte_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[embed_path], opts.dtype, &opts.device)? };
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
            opts.device,
            opts.dtype,
        )?;
        s.tokenizer = Some(tokenizer);
        s.max_seq_len = opts.max_seq_len;
        Ok(s)
    }

    pub fn config(&self) -> &MixtralConfig {
        &self.cfg
    }

    pub fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Greedy / sampled generation. Requires the model was loaded via
    /// `from_pretrained` (so a tokenizer is attached). Mixtral does not
    /// declare an EOS token in the same way Llama does (the field is
    /// optional in MixtralConfig); the loop stops when `max_new_tokens`
    /// is hit.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| crate::error::Error::Msg("StreamingMixtral::generate requires a tokenizer; use from_pretrained".into()))?;
        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(crate::error::Error::Tokenizer)?;
        let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
        if tokens.len() > self.max_seq_len {
            tokens.truncate(self.max_seq_len);
        }

        let mut lp = candle_transformers::generation::LogitsProcessor::new(
            299_792_458,
            Some(temperature).filter(|&t| t > 0.0),
            top_p,
        );
        let mut cache = MixtralStreamCache::new(true, self.dtype, &self.cfg, &self.device)?;

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
        }

        tokenizer
            .decode(&tokens, true)
            .map_err(crate::error::Error::Tokenizer)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut MixtralStreamCache,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut x = self.wte.forward(input_ids)?;
        for (i, path) in self.layer_files.iter().enumerate() {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(path),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrips() {
        // Minimal config that should parse cleanly.
        let j = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "num_experts_per_tok": 2,
            "num_local_experts": 8
        }"#;
        let cfg: MixtralConfig = serde_json::from_str(j).unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_local_experts, 8);
    }
}
