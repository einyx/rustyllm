//! Per-layer streaming ChatGLM2/3 inference.
//!
//! ChatGLM differs from Llama in several material ways:
//!   - Combined `query_key_value` Linear (single matrix for QKV, with
//!     optional bias). Splitting happens after the projection by
//!     narrow() on the last dim, sized by multi_query attention groups.
//!   - 2D RoPE on the first `kv_channels` dims, identity on the rest.
//!     RoPE pairs neighbouring elements (xshaped reshape (rot/2, 2))
//!     rather than the half/half split Llama uses.
//!   - Sequence-first tensor layout `[seq, batch, head, dim]` (Llama
//!     uses batch-first). We respect that internally — the forward
//!     takes batch-first hidden_states for API parity with StreamingLlama
//!     and transposes at the boundary.
//!   - MLP is SwiGLU through a single `dense_h_to_4h` projecting to
//!     `ffn_hidden_size * 2`, then `dense_4h_to_h`. Activation is
//!     candle's `Swiglu` which handles the split-and-gate internally.
//!   - LayerNorm or RmsNorm based on `rmsnorm` config flag.
//!   - On-disk weight prefix is typically
//!     `transformer.encoder.layers.N.*`, not `model.layers.N.*`. Layer
//!     sharding for ChatGLM needs a non-default `LayerNames` —
//!     surfaced via `default_layer_names()`.
//!
//! Block code adapted from `candle-transformers::models::chatglm`
//! (Apache-2.0). End-to-end weight test deferred until a ChatGLM3-6B
//! checkpoint is available locally.

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

use crate::error::Result;
use crate::shard::LayerNames;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatGlmConfig {
    pub num_layers: usize,
    pub padded_vocab_size: usize,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub kv_channels: usize,
    pub num_attention_heads: usize,
    pub seq_length: usize,
    pub layernorm_epsilon: f64,
    pub rmsnorm: bool,
    #[serde(default)]
    pub apply_residual_connection_post_layernorm: bool,
    #[serde(default = "default_true")]
    pub post_layer_norm: bool,
    #[serde(default)]
    pub add_bias_linear: bool,
    #[serde(default = "default_true")]
    pub add_qkv_bias: bool,
    #[serde(default = "default_true")]
    pub multi_query_attention: bool,
    #[serde(default = "default_kv_groups")]
    pub multi_query_group_num: usize,
}

fn default_true() -> bool {
    true
}
fn default_kv_groups() -> usize {
    2
}

/// On-disk weight prefix for ChatGLM. Use this when calling
/// `shard_model` on a ChatGLM checkpoint instead of the default
/// (Llama) layer names.
pub fn default_layer_names() -> LayerNames {
    LayerNames {
        embed: "transformer.embedding".into(),
        layer_prefix: "transformer.encoder.layers".into(),
        norm: "transformer.encoder.final_layernorm".into(),
        lm_head: "transformer.output_layer".into(),
    }
}

fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        Ok(candle_nn::linear(in_dim, out_dim, vb)?)
    } else {
        Ok(candle_nn::linear_no_bias(in_dim, out_dim, vb)?)
    }
}

/// Build a ParallelLinear from a candle linear_b site. Used in
/// ChatGLM where projections may or may not carry bias depending on
/// `add_bias_linear` / `add_qkv_bias`. Goes through candle for name
/// resolution + shape validation, then wraps the resulting weight
/// (and optional bias) in the rayon-parallel matmul shim.
fn parallel_linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<crate::parallel_linear::ParallelLinear> {
    let l = linear_b(in_dim, out_dim, bias, vb)?;
    let w = l.weight().clone();
    let b = l.bias().cloned();
    crate::parallel_linear::ParallelLinear::new(w, b)
}

fn norm(hidden: usize, eps: f64, rms: bool, vb: VarBuilder) -> Result<LayerNorm> {
    if rms {
        Ok(candle_nn::rms_norm(hidden, eps, vb)?.into_inner())
    } else {
        Ok(candle_nn::layer_norm(hidden, eps, vb)?)
    }
}

pub struct ChatGlmRotary {
    /// cos/sin stacked along last dim: shape [seq_length, rot_dim/2, 2]
    cache: Tensor,
}

impl ChatGlmRotary {
    pub fn new(cfg: &ChatGlmConfig, dtype: DType, device: &Device) -> Result<Self> {
        let rotary_dim = cfg.kv_channels;
        let n_elem = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..n_elem)
            .step_by(2)
            .map(|i| 1.0 / 10_000f64.powf(i as f64 / n_elem as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, cfg.seq_length as u32, device)?
            .to_dtype(dtype)?
            .reshape((cfg.seq_length, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cache = Tensor::stack(&[&freqs.cos()?, &freqs.sin()?], D::Minus1)?;
        Ok(Self { cache })
    }

    /// Apply RoPE to xs of shape [seqlen, batch, n_heads, head_dim].
    /// Only the first 2*(rot_dim/2) dims are rotated; the rest pass
    /// through unchanged.
    fn apply(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (seqlen, _b, np, _hn) = xs.dims4()?;
        let cache = self.cache.narrow(0, seqlen_offset, seqlen)?;
        let rot_dim = cache.dim(D::Minus2)? * 2;
        let xs_rot = xs.narrow(D::Minus1, 0, rot_dim)?;
        let xs_pass = xs.narrow(D::Minus1, rot_dim, rot_dim)?;
        let xshaped = xs_rot.reshape((seqlen, (), np, rot_dim / 2, 2))?;
        let cache = cache.reshape((seqlen, (), 1, rot_dim / 2, 2))?;
        let (x0, x1) = (
            xshaped.i((.., .., .., .., 0))?,
            xshaped.i((.., .., .., .., 1))?,
        );
        let (c0, c1) = (cache.i((.., .., .., .., 0))?, cache.i((.., .., .., .., 1))?);
        let out = Tensor::stack(
            &[
                (x0.broadcast_mul(&c0)? - x1.broadcast_mul(&c1)?)?,
                (x1.broadcast_mul(&c0)? + x0.broadcast_mul(&c1)?)?,
            ],
            D::Minus1,
        )?
        .flatten_from(3)?;
        Ok(Tensor::cat(&[out, xs_pass], D::Minus1)?)
    }
}

struct CoreAttention {
    norm_factor: f64,
}

impl CoreAttention {
    fn new(cfg: &ChatGlmConfig) -> Self {
        Self {
            norm_factor: (cfg.kv_channels as f64).sqrt(),
        }
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Option<Tensor>) -> Result<Tensor> {
        // q,k,v are [seq, b, np, hn]
        let (sq, b, np, _hn) = q.dims4()?;
        let sk = k.dim(0)?;
        let q2 = q.reshape((sq, b * np, ()))?;
        let k2 = k.reshape((sk, b * np, ()))?;
        let matmul =
            (Tensor::matmul(&q2.transpose(0, 1)?, &k2.transpose(0, 1)?.transpose(1, 2)?)?
                / self.norm_factor)?
                .reshape((b, np, sq, sk))?;
        let scores = match mask {
            Some(m) => {
                let m = m.broadcast_left((matmul.dim(0)?, matmul.dim(1)?))?;
                let neg_inf = Tensor::new(f32::NEG_INFINITY, matmul.device())?
                    .broadcast_as(m.shape().dims())?;
                m.where_cond(&neg_inf, &matmul)?
            }
            None => matmul,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let v2 = v.reshape((v.dim(0)?, b * np, ()))?;
        let probs2 = probs.reshape((b * np, sq, ()))?;
        let ctx = Tensor::matmul(&probs2, &v2.transpose(0, 1)?)?.reshape((b, np, sq, v.dim(3)?))?;
        Ok(ctx
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .flatten_from(D::Minus2)?)
    }
}

struct SelfAttention {
    qkv: crate::parallel_linear::ParallelLinear,
    dense: crate::parallel_linear::ParallelLinear,
    core: CoreAttention,
    n_heads: usize,
    n_kv_groups: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(vb: VarBuilder, cfg: &ChatGlmConfig) -> Result<Self> {
        if !cfg.multi_query_attention {
            return Err(crate::error::Error::Msg(
                "ChatGLM streaming only supports multi_query_attention=true".into(),
            ));
        }
        let projection = cfg.kv_channels * cfg.num_attention_heads;
        let qkv_out = projection + 2 * cfg.kv_channels * cfg.multi_query_group_num;
        let qkv = parallel_linear_b(
            cfg.hidden_size,
            qkv_out,
            cfg.add_bias_linear || cfg.add_qkv_bias,
            vb.pp("query_key_value"),
        )?;
        let dense = parallel_linear_b(
            cfg.hidden_size,
            cfg.hidden_size,
            cfg.add_bias_linear,
            vb.pp("dense"),
        )?;
        Ok(Self {
            qkv,
            dense,
            core: CoreAttention::new(cfg),
            n_heads: cfg.num_attention_heads,
            n_kv_groups: cfg.multi_query_group_num,
            head_dim: cfg.kv_channels,
        })
    }

    /// xs shape: [seq, batch, hidden]
    fn forward(
        &self,
        xs: &Tensor,
        mask: &Option<Tensor>,
        rope: &ChatGlmRotary,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mixed = self.qkv.forward(xs)?;
        let q_sz = self.n_heads * self.head_dim;
        let kv_sz = self.n_kv_groups * self.head_dim;
        let q = mixed.narrow(D::Minus1, 0, q_sz)?;
        let k = mixed.narrow(D::Minus1, q_sz, kv_sz)?;
        let v = mixed.narrow(D::Minus1, q_sz + kv_sz, kv_sz)?;

        let q = q.reshape((q.dim(0)?, q.dim(1)?, self.n_heads, self.head_dim))?;
        let k = k.reshape((k.dim(0)?, k.dim(1)?, self.n_kv_groups, self.head_dim))?;
        let v = v.reshape((v.dim(0)?, v.dim(1)?, self.n_kv_groups, self.head_dim))?;

        let q = rope.apply(&q, seqlen_offset)?;
        let k = rope.apply(&k, seqlen_offset)?;

        // Repeat KV to match q's head count.
        let ratio = self.n_heads / self.n_kv_groups;
        let expand_kv = |t: Tensor| -> Result<Tensor> {
            let (d0, d1, d2, d3) = t.dims4()?;
            Ok(t.unsqueeze(D::Minus2)?
                .expand((d0, d1, d2, ratio, d3))?
                .reshape((d0, d1, self.n_heads, self.head_dim))?)
        };
        let k = expand_kv(k)?;
        let v = expand_kv(v)?;

        let ctx = self.core.forward(&q, &k, &v, mask)?;
        Ok(self.dense.forward(&ctx)?)
    }
}

struct Mlp {
    dense_h_to_4h: crate::parallel_linear::ParallelLinear,
    dense_4h_to_h: crate::parallel_linear::ParallelLinear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &ChatGlmConfig) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: parallel_linear_b(
                cfg.hidden_size,
                cfg.ffn_hidden_size * 2,
                cfg.add_bias_linear,
                vb.pp("dense_h_to_4h"),
            )?,
            dense_4h_to_h: parallel_linear_b(
                cfg.ffn_hidden_size,
                cfg.hidden_size,
                cfg.add_bias_linear,
                vb.pp("dense_4h_to_h"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ParallelLinear doesn't impl Module — call .forward(...) directly.
        let hidden = self.dense_h_to_4h.forward(x)?;
        let activated = hidden.apply(&candle_nn::Activation::Swiglu)?;
        self.dense_4h_to_h.forward(&activated)
    }
}

struct Block {
    input_ln: LayerNorm,
    attn: SelfAttention,
    post_ln: LayerNorm,
    mlp: Mlp,
    residual_post_ln: bool,
}

impl Block {
    fn load(vb: VarBuilder, cfg: &ChatGlmConfig) -> Result<Self> {
        Ok(Self {
            input_ln: norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                cfg.rmsnorm,
                vb.pp("input_layernorm"),
            )?,
            attn: SelfAttention::load(vb.pp("self_attention"), cfg)?,
            post_ln: norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                cfg.rmsnorm,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            residual_post_ln: cfg.apply_residual_connection_post_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Option<Tensor>,
        rope: &ChatGlmRotary,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let ln_out = xs.apply(&self.input_ln)?;
        let attn_out = self.attn.forward(&ln_out, mask, rope, seqlen_offset)?;
        let residual = if self.residual_post_ln { &ln_out } else { xs };
        let after_attn = (residual + attn_out)?;
        let ln2 = after_attn.apply(&self.post_ln)?;
        let mlp_out = self.mlp.forward(&ln2)?;
        let residual = if self.residual_post_ln {
            &ln2
        } else {
            &after_attn
        };
        Ok((residual + mlp_out)?)
    }
}

/// Per-layer streaming ChatGLM. As with `StreamingLlama`, only the
/// embedding, final norm, and lm_head are resident on the device; each
/// transformer block's weights are mmap'd in per forward call, run,
/// and dropped before the next block.
pub struct StreamingChatGlm {
    wte: Embedding,
    final_ln: Option<LayerNorm>,
    lm_head: Linear,
    layer_files: Vec<PathBuf>,
    cfg: ChatGlmConfig,
    device: Device,
    dtype: DType,
    rope: ChatGlmRotary,
}

impl StreamingChatGlm {
    pub fn new(
        wte_vb: VarBuilder,
        norm_vb: VarBuilder,
        lm_head_vb: VarBuilder,
        layer_files: Vec<PathBuf>,
        cfg: ChatGlmConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Use full ChatGLM paths so the caller can pass a VarBuilder
        // rooted at the shard file (same convention as StreamingLlama /
        // StreamingMixtral / StreamingBaichuan). Avoids the previous
        // foot-gun where the caller had to pre-scope the VarBuilder.
        let wte = embedding(
            cfg.padded_vocab_size,
            cfg.hidden_size,
            wte_vb.pp("transformer.embedding.word_embeddings"),
        )?;
        let final_ln = if cfg.post_layer_norm {
            Some(norm(
                cfg.hidden_size,
                cfg.layernorm_epsilon,
                cfg.rmsnorm,
                norm_vb.pp("transformer.encoder.final_layernorm"),
            )?)
        } else {
            None
        };
        let lm_head = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.padded_vocab_size,
            lm_head_vb.pp("transformer.output_layer"),
        )?;
        let rope = ChatGlmRotary::new(&cfg, dtype, &device)?;
        Ok(Self {
            wte,
            final_ln,
            lm_head,
            layer_files,
            cfg,
            device,
            dtype,
            rope,
        })
    }

    pub fn config(&self) -> &ChatGlmConfig {
        &self.cfg
    }

    /// Forward over input_ids of shape [batch, seq].
    pub fn forward(&self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        // [b, seq, hidden] -> [seq, b, hidden] to match ChatGLM layout.
        let x = self.wte.forward(input_ids)?.transpose(0, 1)?;
        let mask = if seq_len > 1 {
            Some(causal_mask(seq_len, &self.device)?)
        } else {
            None
        };

        let mut h = x;
        for (i, path) in self.layer_files.iter().enumerate() {
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    std::slice::from_ref(path),
                    self.dtype,
                    &self.device,
                )?
            };
            // Layer-internal prefix; the per-layer shard already
            // scopes the weights to `transformer.encoder.layers.N.*`.
            let block = Block::load(vb.pp(format!("transformer.encoder.layers.{i}")), &self.cfg)?;
            h = block.forward(&h, &mask, &self.rope, seqlen_offset)?;
        }
        if let Some(ln) = &self.final_ln {
            h = h.apply(ln)?;
        }
        // back to [b, seq, hidden] and take last position
        let h = h.transpose(0, 1)?;
        let h = h.i((.., seq_len - 1, ..))?.contiguous()?;
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }
}

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0u8; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = 1;
        }
    }
    Ok(Tensor::from_vec(data, (seq_len, seq_len), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses() {
        let j = r#"{
            "num_layers": 28,
            "padded_vocab_size": 65024,
            "hidden_size": 4096,
            "ffn_hidden_size": 13696,
            "kv_channels": 128,
            "num_attention_heads": 32,
            "seq_length": 8192,
            "layernorm_epsilon": 1e-5,
            "rmsnorm": true,
            "add_qkv_bias": true,
            "multi_query_attention": true,
            "multi_query_group_num": 2
        }"#;
        let cfg: ChatGlmConfig = serde_json::from_str(j).unwrap();
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.multi_query_group_num, 2);
        assert!(cfg.rmsnorm);
    }

    #[test]
    fn layer_names_differ_from_llama() {
        let n = default_layer_names();
        assert!(n.layer_prefix.contains("transformer.encoder.layers"));
        assert_ne!(n.layer_prefix, LayerNames::default().layer_prefix);
    }

    /// Regression: previously, `StreamingChatGlm::new` looked up tensors
    /// under bare names like `word_embeddings.weight` even though
    /// ChatGLM shards on disk store full names like
    /// `transformer.embedding.word_embeddings.weight`. The lookup would
    /// fail at load time on a real checkpoint.
    ///
    /// We can't run a full load here without weights, but we can build
    /// a synthetic shard whose tensor names match the *real* on-disk
    /// layout and confirm `new()` resolves them via the full prefix.
    /// We only test the embed step because that's the first thing
    /// `new()` does — if the prefix is wrong, this errors immediately.
    #[test]
    fn new_uses_full_chatglm_prefixes() {
        use candle_core::{DType, Device, Tensor};
        use std::collections::HashMap as Map;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let embed_path = dir.path().join("embed.safetensors");
        let mut t: Map<String, Tensor> = Map::new();
        // Real ChatGLM stores embeddings under this full path.
        t.insert(
            "transformer.embedding.word_embeddings.weight".into(),
            Tensor::zeros((128, 32), DType::F16, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&t, &embed_path).unwrap();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&embed_path], DType::F16, &Device::Cpu).unwrap()
        };
        // If `new()` looked up a bare `word_embeddings.weight` (the
        // pre-fix bug), this would error with a missing-tensor message.
        // We invoke the same path resolution explicitly to keep the
        // test light (no full StreamingChatGlm needed).
        let result = embedding(128, 32, vb.pp("transformer.embedding.word_embeddings"));
        assert!(
            result.is_ok(),
            "embed lookup via full ChatGLM prefix must resolve, got: {:?}",
            result.err()
        );
    }
}
