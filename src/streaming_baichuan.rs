//! Per-layer streaming Baichuan-13B inference.
//!
//! Baichuan-13B differs from Llama in two material ways:
//!   - Combined `W_pack` Linear (single matrix projecting hidden_size
//!     -> 3 * hidden_size for QKV jointly), then split by narrow().
//!   - ALiBi (Attention with Linear Biases) instead of RoPE. A
//!     precomputed bias tensor of shape [n_heads, 1, seq_len] is added
//!     to attention scores before softmax, encoding relative position
//!     as a linear penalty. Each head gets its own slope.
//!
//! Otherwise structurally Llama-like: RmsNorm, SwiGLU MLP, residual
//! attention + MLP. Baichuan-7B uses RoPE and is Llama-compatible
//! (handled by [`crate::streaming::StreamingLlama`]); use this module
//! only for the 13B (ALiBi) variant.
//!
//! End-to-end weight test deferred until a Baichuan-13B checkpoint is
//! available locally.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, linear_no_bias as linear, Activation, Embedding, Linear, RmsNorm, VarBuilder,
};
use serde::Deserialize;

use crate::error::Result;
use crate::parallel_linear::ParallelLinear;

fn parallel_linear_no_bias(
    in_features: usize,
    out_features: usize,
    vb: VarBuilder,
) -> Result<ParallelLinear> {
    let l = linear(in_features, out_features, vb)?;
    ParallelLinear::new(l.weight().clone(), None)
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaichuanConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_act() -> Activation {
    Activation::Silu
}

impl BaichuanConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// ALiBi slopes per head, following the original "Train Short, Test
/// Long" paper (Press et al.). For n heads, slope_i = 2^(-8 * (i+1) /
/// n). When n is not a power of 2, we interpolate the standard pattern
/// — but Baichuan-13B has n=40 which is not a power of 2, so we use
/// the same fallback HF transformers uses (closest power of 2 + extra
/// slopes from the next power up).
fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    // Power-of-2 case: slopes[i] = start^(i+1), where start = 2^(-8/n).
    // Equivalently start * start^i. Matches HF transformers reference.
    fn power_of_two_slopes(n: usize) -> Vec<f32> {
        let start = 2f64.powf(-(8.0 / n as f64)) as f32;
        let mut out = Vec::with_capacity(n);
        let mut cur = start;
        for _ in 0..n {
            out.push(cur);
            cur *= start;
        }
        out
    }
    if num_heads.is_power_of_two() {
        return power_of_two_slopes(num_heads);
    }
    // Non-power-of-2: HF takes the slopes for the nearest lower power of 2,
    // then appends every-other slope from the next power up, taking only as
    // many as needed to reach num_heads. This is not strictly monotonic
    // across the join (the extras start at higher values than the tail of
    // the base set) but matches reference Baichuan-13B behavior.
    let closest = num_heads.next_power_of_two() / 2;
    let mut slopes = power_of_two_slopes(closest);
    let extra: Vec<f32> = power_of_two_slopes(2 * closest)
        .into_iter()
        .step_by(2)
        .take(num_heads - closest)
        .collect();
    slopes.extend(extra);
    slopes
}

/// Build ALiBi bias of shape [n_heads, 1, seq_len] used by attention.
/// The full causal bias is `(j - i) * slope_h` for query position i,
/// key position j. We materialize the row-vector form (j) and let it
/// broadcast against [seq_len, seq_len] when the mask is applied.
fn alibi_bias(num_heads: usize, seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let slopes = alibi_slopes(num_heads);
    let slopes = Tensor::from_vec(slopes, (num_heads, 1, 1), device)?.to_dtype(dtype)?;
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (1, 1, seq_len), device)?.to_dtype(dtype)?;
    Ok(slopes.broadcast_mul(&positions)?)
}

pub struct BaichuanCache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    alibi_cache: Tensor,
    device: Device,
}

impl BaichuanCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        cfg: &BaichuanConfig,
        device: &Device,
    ) -> Result<Self> {
        let alibi = alibi_bias(
            cfg.num_attention_heads,
            cfg.max_position_embeddings,
            device,
            dtype,
        )?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; cfg.num_hidden_layers],
            alibi_cache: alibi,
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

    fn alibi_slice(&self, kv_len: usize) -> Result<Tensor> {
        // [n_heads, 1, kv_len]
        Ok(self.alibi_cache.narrow(D::Minus1, 0, kv_len)?)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    Ok(mask.where_cond(&on_true, on_false)?)
}

struct Attention {
    w_pack: ParallelLinear,
    o_proj: ParallelLinear,
    n_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &BaichuanConfig) -> Result<Self> {
        Ok(Self {
            w_pack: parallel_linear_no_bias(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("W_pack"))?,
            o_proj: parallel_linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut BaichuanCache,
    ) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;
        let mixed = self.w_pack.forward(x)?;
        let h = self.hidden_size;
        let q = mixed.narrow(D::Minus1, 0, h)?;
        let k = mixed.narrow(D::Minus1, h, h)?;
        let v = mixed.narrow(D::Minus1, 2 * h, h)?;

        // [b, seq, n_heads, head_dim] -> [b, n_heads, seq, head_dim]
        let q = q
            .reshape((b, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        if cache.use_kv_cache {
            if let Some((ck, cv)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[ck, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cv, &v], 2)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let kv_len = k.dim(2)?;
        let q_f = q.to_dtype(DType::F32)?;
        let k_f = k.to_dtype(DType::F32)?;
        let v_f = v.to_dtype(DType::F32)?;
        let mut scores = (q_f.matmul(&k_f.t()?)? / (self.head_dim as f64).sqrt())?;

        // ALiBi: add per-head linear bias on key position. Shape:
        // [n_heads, 1, kv_len], broadcasts to [b, n_heads, seq, kv_len].
        let alibi = cache
            .alibi_slice(kv_len)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?; // [1, n_heads, 1, kv_len]
        scores = scores.broadcast_add(&alibi)?;

        if seq_len > 1 {
            let mask = cache
                .mask(seq_len, index_pos)?
                .broadcast_as(scores.shape())?;
            scores = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let y = probs.matmul(&v_f.contiguous()?)?.to_dtype(x.dtype())?;
        let y = y
            .transpose(1, 2)?
            .reshape(&[b, seq_len, self.hidden_size])?;
        Ok(self.o_proj.forward(&y)?)
    }
}

struct Mlp {
    gate: ParallelLinear,
    up: ParallelLinear,
    down: ParallelLinear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &BaichuanConfig) -> Result<Self> {
        Ok(Self {
            gate: parallel_linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: parallel_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: parallel_linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        Ok(self.down.forward(&y)?)
    }
}

struct Block {
    input_ln: RmsNorm,
    attn: Attention,
    post_ln: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &BaichuanConfig) -> Result<Self> {
        Ok(Self {
            input_ln: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attn: Attention::load(vb.pp("self_attn"), cfg)?,
            post_ln: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut BaichuanCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.input_ln.forward(x)?;
        let h = (self.attn.forward(&h, index_pos, block_idx, cache)? + residual)?;
        let residual = &h;
        let h = (self.mlp.forward(&self.post_ln.forward(&h)?)? + residual)?;
        Ok(h)
    }
}

pub struct StreamingBaichuan {
    wte: Embedding,
    ln_f: RmsNorm,
    lm_head: Linear,
    layer_files: Vec<PathBuf>,
    cfg: BaichuanConfig,
    device: Device,
    dtype: DType,
}

impl StreamingBaichuan {
    pub fn new(
        wte_vb: VarBuilder,
        norm_vb: VarBuilder,
        lm_head_vb: VarBuilder,
        layer_files: Vec<PathBuf>,
        cfg: BaichuanConfig,
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
        })
    }

    pub fn config(&self) -> &BaichuanConfig {
        &self.cfg
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut BaichuanCache,
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
        Ok(self.lm_head.forward(&x)?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses() {
        let j = r#"{
            "vocab_size": 64000,
            "hidden_size": 5120,
            "intermediate_size": 13696,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6
        }"#;
        let cfg: BaichuanConfig = serde_json::from_str(j).unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.num_attention_heads, 40);
    }

    #[test]
    fn alibi_slopes_power_of_two() {
        let s = alibi_slopes(8);
        assert_eq!(s.len(), 8);
        // First slope = 2^(-8/8) = 0.5
        assert!((s[0] - 0.5).abs() < 1e-6);
        // Each subsequent multiplies by start = 0.5
        assert!((s[1] - 0.25).abs() < 1e-6);
        assert!((s[7] - (1.0 / 256.0)).abs() < 1e-6);
    }

    #[test]
    fn alibi_slopes_non_power_of_two() {
        // Baichuan-13B has 40 heads (not a power of 2). HF's fallback
        // formula isn't strictly monotonic at the join — we only check
        // the right count, all values in (0, 1), and monotonicity
        // within each contiguous source (first 32 from base, last 8
        // from interleaved next-power-up).
        let s = alibi_slopes(40);
        assert_eq!(s.len(), 40);
        for v in &s {
            assert!(*v > 0.0 && *v < 1.0, "slope out of range: {v}");
        }
        for w in s[..32].windows(2) {
            assert!(w[1] < w[0], "base slopes must decrease");
        }
        for w in s[32..].windows(2) {
            assert!(w[1] < w[0], "extra slopes must decrease");
        }
    }
}
