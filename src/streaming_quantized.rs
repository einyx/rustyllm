//! Per-layer streaming Llama inference over quantized GGUF shards.
//!
//! Mirrors [`crate::streaming::StreamingLlama`] but loads per-layer
//! weights from GGUF files produced by [`crate::quantize`] instead of
//! mmapped F16 safetensors. Each Linear is replaced by `QMatMul`,
//! which dequantizes a block at a time during matmul. Saves ~4x disk
//! I/O and device residency per layer with negligible quality loss
//! for Q4_K_M / Q4_0 / Q8_0.
//!
//! Embeddings, final norm, and lm_head are kept F16 in the GGUF (per
//! [`crate::quantize::should_quantize`]), so they load as normal
//! tensors via QTensor::dequantize.
//!
//! Block code mirrors `streaming.rs` exactly; only the Linear → QMatMul
//! swap differs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Embedding, RmsNorm};
use candle_transformers::models::llama::Config;
use candle_transformers::utils::repeat_kv;

use crate::error::{Error, Result};

/// Per-layer GGUF reader. Slurps the whole file into a buffer at
/// construction so all subsequent tensor reads are zero-syscall in-
/// memory slices. On RAM-tight hosts this is offset by an explicit
/// `release_buffer()` after weights are copied into device memory.
///
/// Why slurp + Cursor (vs File + seek): GGUF tensor data is laid out
/// roughly sequentially in the file, but the gguf reader issues
/// individual seek+read calls per tensor (~9 per layer for Q4K). Each
/// seek can break SSD read-ahead, dropping effective throughput. One
/// big sequential read uses the SSD's full bandwidth, then random-
/// access slicing on the in-memory buffer is free.
pub(crate) struct GgufLayer {
    content: gguf_file::Content,
    buffer: Vec<u8>,
    device: Device,
}

impl GgufLayer {
    pub(crate) fn open(path: &Path, device: &Device) -> Result<Self> {
        let mut file = File::open(path)?;
        let len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        let mut buffer = Vec::with_capacity(len);
        file.read_to_end(&mut buffer)?;
        let mut cursor = Cursor::new(&buffer[..]);
        let content = gguf_file::Content::read(&mut cursor)?;
        Ok(Self {
            content,
            buffer,
            device: device.clone(),
        })
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let mut cursor = Cursor::new(&self.buffer[..]);
        let qt = self
            .content
            .tensor(&mut cursor, name, &self.device)
            .map_err(|e| Error::Msg(format!("gguf tensor {name}: {e}")))?;
        Ok(QMatMul::from_arc(Arc::new(qt))?)
    }

    /// Dequantize a tensor (used for norms / embeddings stored as F16
    /// inside the GGUF).
    fn tensor(&mut self, name: &str) -> Result<Tensor> {
        let mut cursor = Cursor::new(&self.buffer[..]);
        let qt = self
            .content
            .tensor(&mut cursor, name, &self.device)
            .map_err(|e| Error::Msg(format!("gguf tensor {name}: {e}")))?;
        Ok(qt.dequantize(&self.device)?)
    }

    /// Free the in-memory buffer. Call after all tensors have been
    /// loaded (QBlock holds owned tensor data, so the buffer is no
    /// longer referenced). Releases ~500 MB per layer back to the
    /// allocator immediately, keeping peak RSS bounded during the
    /// streaming forward.
    pub(crate) fn release_buffer(&mut self) {
        self.buffer = Vec::new();
    }
}

pub struct QuantStreamCache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl QuantStreamCache {
    pub fn new(use_kv_cache: bool, dtype: DType, cfg: &Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(inv_freq, device)?;
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

    /// Drop KV cache entries past `new_len` for every layer. Used by
    /// speculative decoding to roll back tokens the verifier rejected:
    /// the draft + verify pass advances the cache by `k+1` positions,
    /// but we only commit the accepted prefix. The KV tensor shape is
    /// `(batch, n_kv_heads, seq_len, head_dim)`, so we narrow on dim 2.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        for kv in self.kvs.iter_mut() {
            if let Some((k, v)) = kv.as_mut() {
                if k.dims()[2] > new_len {
                    *k = k.narrow(2, 0, new_len)?.contiguous()?;
                    *v = v.narrow(2, 0, new_len)?.contiguous()?;
                }
            }
        }
        Ok(())
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

pub(crate) struct QAttn {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_pos: usize,
}

impl QAttn {
    fn load(layer: &mut GgufLayer, layer_idx: usize, cfg: &Config) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}.self_attn");
        Ok(Self {
            q_proj: layer.qmatmul(&format!("{p}.q_proj.weight"))?,
            k_proj: layer.qmatmul(&format!("{p}.k_proj.weight"))?,
            v_proj: layer.qmatmul(&format!("{p}.v_proj.weight"))?,
            o_proj: layer.qmatmul(&format!("{p}.o_proj.weight"))?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_pos: cfg.max_position_embeddings,
        })
    }

    fn apply_rope(&self, x: &Tensor, index_pos: usize, cache: &QuantStreamCache) -> Result<Tensor> {
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
        cache: &mut QuantStreamCache,
    ) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b, seq_len, self.n_kv_heads, self.head_dim))?
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

        let k = repeat_kv(k, self.n_heads / self.n_kv_heads)?;
        let v = repeat_kv(v, self.n_heads / self.n_kv_heads)?;

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

pub(crate) struct QMlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl QMlp {
    fn load(layer: &mut GgufLayer, layer_idx: usize) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}.mlp");
        Ok(Self {
            gate: layer.qmatmul(&format!("{p}.gate_proj.weight"))?,
            up: layer.qmatmul(&format!("{p}.up_proj.weight"))?,
            down: layer.qmatmul(&format!("{p}.down_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        Ok(self.down.forward(&y)?)
    }
}

pub(crate) struct QBlock {
    rms_1: RmsNorm,
    attn: QAttn,
    rms_2: RmsNorm,
    mlp: QMlp,
}

impl QBlock {
    pub(crate) fn load(layer: &mut GgufLayer, layer_idx: usize, cfg: &Config) -> Result<Self> {
        let p = format!("model.layers.{layer_idx}");
        // Norms live in the GGUF as F16; dequantize gives F16/F32 tensors.
        let rms_1_w = layer.tensor(&format!("{p}.input_layernorm.weight"))?;
        let rms_2_w = layer.tensor(&format!("{p}.post_attention_layernorm.weight"))?;
        let attn = QAttn::load(layer, layer_idx, cfg)?;
        let mlp = QMlp::load(layer, layer_idx)?;
        Ok(Self {
            rms_1: RmsNorm::new(rms_1_w, cfg.rms_norm_eps),
            attn,
            rms_2: RmsNorm::new(rms_2_w, cfg.rms_norm_eps),
            mlp,
        })
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut QuantStreamCache,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.rms_1.forward(x)?;
        let h = (self.attn.forward(&h, index_pos, block_idx, cache)? + residual)?;
        let residual = &h;
        Ok((self.mlp.forward(&self.rms_2.forward(&h)?)? + residual)?)
    }
}

/// Per-layer streaming Llama over quantized GGUF shards.
pub struct StreamingLlamaQuantized {
    wte: Embedding,
    ln_f: RmsNorm,
    lm_head: candle_nn::Linear,
    layer_files: Vec<PathBuf>,
    cfg: Config,
    device: Device,
    tokenizer: Option<tokenizers::Tokenizer>,
    max_seq_len: usize,
    /// Pre-loaded blocks for layers `[0, resident_blocks.len())`. Stays
    /// on the device across forward calls — eliminates the per-token
    /// per-layer GGUF read + QMatMul construction overhead, at the cost
    /// of holding ~400 MB of GPU memory per resident layer (Q4 70B).
    /// Layers `[resident_blocks.len(), layer_files.len())` are still
    /// streamed every token.
    resident_blocks: Vec<QBlock>,
    /// CPU-resident blocks: same Q4 weights held in host RAM (one
    /// per layer here). At forward time the hidden state is transferred
    /// to CPU, these run, then the result transfers back. Cuts the
    /// per-streamed-layer overhead (~200 ms on GPU) to candle's CPU
    /// QMatMul cost (~8 ms / layer / token), at the price of two
    /// host↔device transfers per token. The hybrid is the right move
    /// when more layers don't fit on the GPU but do fit in host RAM.
    cpu_resident_blocks: Vec<QBlock>,
    /// Persistent KV cache for the CPU-resident layers. Must live
    /// across forward() calls or the CPU layers see no history past
    /// the current token and the model drifts immediately. Wrapped in
    /// `Mutex<Option<>>` so we can mutate it through the `&self`
    /// forward signature (interior mutability, single-writer at a
    /// time which matches our generate loop).
    cpu_cache: std::sync::Mutex<Option<QuantStreamCache>>,
}

impl StreamingLlamaQuantized {
    /// Build from resident weights + per-layer GGUF paths.
    ///
    /// `embed_path` should be a GGUF containing `model.embed_tokens.weight`;
    /// `norm_path` should be a GGUF with `model.norm.weight`;
    /// `lm_head_path` should be a GGUF with `lm_head.weight`.
    /// Each transformer-layer GGUF must contain the full set of
    /// `model.layers.{i}.*` weights produced by [`crate::quantize`].
    pub fn new(
        embed_path: &Path,
        norm_path: &Path,
        lm_head_path: &Path,
        layer_files: Vec<PathBuf>,
        cfg: Config,
        device: Device,
    ) -> Result<Self> {
        let mut e = GgufLayer::open(embed_path, &device)?;
        let embed = e.tensor("model.embed_tokens.weight")?;
        let wte = Embedding::new(embed.clone(), cfg.hidden_size);

        let mut n = GgufLayer::open(norm_path, &device)?;
        let ln_f = RmsNorm::new(n.tensor("model.norm.weight")?, cfg.rms_norm_eps);

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed, None)
        } else {
            let mut lh = GgufLayer::open(lm_head_path, &device)?;
            candle_nn::Linear::new(lh.tensor("lm_head.weight")?, None)
        };

        Ok(Self {
            wte,
            ln_f,
            lm_head,
            layer_files,
            cfg,
            device,
            tokenizer: None,
            max_seq_len: 512,
            resident_blocks: Vec::new(),
            cpu_resident_blocks: Vec::new(),
            cpu_cache: std::sync::Mutex::new(None),
        })
    }

    /// Pre-load `n` layers' weights and keep them resident on the
    /// device across forward calls. Subsequent `forward` will use these
    /// for layers `[0, n)` and stream the rest as usual.
    ///
    /// Bounded by `layer_files.len()`. Each Q4 layer is ~400 MB GPU.
    /// On a 12 GB GPU with Q4 70B, n ≈ 20 fits comfortably alongside
    /// activations and the per-token kv cache.
    pub fn pin_resident_layers(&mut self, n: usize) -> Result<()> {
        let n = n.min(self.layer_files.len());
        self.resident_blocks.clear();
        self.resident_blocks.reserve(n);
        for i in 0..n {
            let mut layer = GgufLayer::open(&self.layer_files[i], &self.device)?;
            let block = QBlock::load(&mut layer, i, &self.cfg)?;
            self.resident_blocks.push(block);
        }
        Ok(())
    }

    /// Pre-load `n` layers into host RAM as CPU-device QBlocks. These
    /// run after the GPU-resident layers and before disk streaming:
    /// hidden state transfers GPU→CPU, runs on CPU candle QMatMul,
    /// then transfers back GPU. Use when more layers fit in host RAM
    /// than on the GPU.
    ///
    /// Layer indices used: `[gpu_resident_count, gpu_resident_count + n)`.
    pub fn pin_cpu_resident_layers(&mut self, n: usize) -> Result<()> {
        let gpu_count = self.resident_blocks.len();
        let max = self.layer_files.len().saturating_sub(gpu_count);
        let n = n.min(max);
        self.cpu_resident_blocks.clear();
        self.cpu_resident_blocks.reserve(n);
        for offset in 0..n {
            let i = gpu_count + offset;
            let mut layer = GgufLayer::open(&self.layer_files[i], &Device::Cpu)?;
            let block = QBlock::load(&mut layer, i, &self.cfg)?;
            self.cpu_resident_blocks.push(block);
        }
        Ok(())
    }

    /// Load from a HuggingFace model dir + a directory of pre-quantized
    /// per-layer GGUF shards. Mirrors `StreamingLlama::from_pretrained`
    /// but pulls weights from `.gguf` instead of `.safetensors` so disk
    /// reads are ~4× smaller — the deciding factor when the F16 working
    /// set doesn't fit in the OS page cache.
    ///
    /// `model_id` is used only to fetch `config.json` + `tokenizer.json`
    /// (small, no weight download). `quantized_shards_dir` must contain
    /// the GGUF files produced by [`crate::quantize::quantize_split_directory`]
    /// (one per layer + embed/norm/lm_head).
    pub fn from_pretrained(
        model_id: &str,
        quantized_shards_dir: &Path,
        opts: crate::inference::LoadOptions,
    ) -> Result<Self> {
        let (model_dir, _) =
            crate::inference::resolve_model_dir_metadata_only(model_id, opts.hf_token.as_deref())?;
        let cfg = crate::family::load_llama_compatible_config(&model_dir)?.ok_or_else(|| {
            Error::Msg(format!(
                "model at {} uses a non-Llama-compatible architecture",
                model_dir.display()
            ))
        })?;
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(Error::Tokenizer)?;

        // Discover per-layer GGUF files in numeric order.
        let mut layer_files: Vec<(usize, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(quantized_shards_dir)? {
            let entry = entry?;
            let name = match entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            // model.layers.N.gguf → N
            if let Some(rest) = name.strip_prefix("model.layers.") {
                if let Some(idx_str) = rest.strip_suffix(".gguf") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        layer_files.push((idx, entry.path()));
                    }
                }
            }
        }
        layer_files.sort_by_key(|(i, _)| *i);
        let layer_files: Vec<PathBuf> = layer_files.into_iter().map(|(_, p)| p).collect();
        if layer_files.len() != cfg.num_hidden_layers {
            return Err(Error::Msg(format!(
                "expected {} layer GGUFs in {}, found {}",
                cfg.num_hidden_layers,
                quantized_shards_dir.display(),
                layer_files.len()
            )));
        }

        let embed_path = quantized_shards_dir.join("model.embed_tokens.gguf");
        let norm_path = quantized_shards_dir.join("model.norm.gguf");
        let lm_head_path = quantized_shards_dir.join("lm_head.gguf");

        let mut s = Self::new(
            &embed_path,
            &norm_path,
            &lm_head_path,
            layer_files,
            cfg,
            opts.device,
        )?;
        s.tokenizer = Some(tokenizer);
        s.max_seq_len = opts.max_seq_len;
        // Partial-residency: env var RUSTYLLM_RESIDENT_LAYERS=N keeps
        // the first N layers permanently on the device. Per-token
        // streaming overhead drops proportionally; peak GPU memory
        // grows by ~one_layer_size * N (~400 MB for Q4 70B).
        if let Ok(s_val) = std::env::var("RUSTYLLM_RESIDENT_LAYERS") {
            if let Ok(n) = s_val.parse::<usize>() {
                s.pin_resident_layers(n)?;
            }
        }
        // RUSTYLLM_CPU_RESIDENT_LAYERS: number of layers (starting
        // after the GPU-resident set) to keep in host RAM as CPU
        // QBlocks. Trades two host↔device transfers per token for
        // eliminating the ~200ms-per-layer GPU streaming overhead.
        if let Ok(s_val) = std::env::var("RUSTYLLM_CPU_RESIDENT_LAYERS") {
            if let Ok(n) = s_val.parse::<usize>() {
                s.pin_cpu_resident_layers(n)?;
            }
        }
        Ok(s)
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Greedy / sampled text generation. Requires the model was loaded
    /// via `from_pretrained` (so a tokenizer is attached).
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<String> {
        use candle_transformers::generation::LogitsProcessor;
        use candle_transformers::models::llama::LlamaEosToks;

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            Error::Msg(
                "StreamingLlamaQuantized::generate requires tokenizer; use from_pretrained".into(),
            )
        })?;

        let encoded = tokenizer.encode(prompt, true).map_err(Error::Tokenizer)?;
        let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
        if tokens.len() > self.max_seq_len {
            tokens.truncate(self.max_seq_len);
        }

        let mut lp =
            LogitsProcessor::new(299_792_458, Some(temperature).filter(|&t| t > 0.0), top_p);
        // QMatMul dequantizes Q4 weights into F32 activations, so cos/sin
        // must also be F32 to satisfy rope's same-dtype invariant.
        let mut cache = QuantStreamCache::new(true, DType::F32, &self.cfg, &self.device)?;

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

        tokenizer.decode(&tokens, true).map_err(Error::Tokenizer)
    }

    /// Like `forward`, but returns logits at every input position instead
    /// of only the last one. Shape: `(batch, seq_len, vocab)`. Needed by
    /// speculative decoding so the verifier can score each draft token
    /// individually in one batched call.
    /// Truncate every KV cache the model touches per forward — the
    /// per-call `cache` arg AND the internal `cpu_cache` used by the
    /// CPU residency tier. Speculative decoding has to roll back both
    /// or the CPU-tier layers desync from the GPU-tier layers.
    pub fn truncate_caches(&self, cache: &mut QuantStreamCache, new_len: usize) -> Result<()> {
        cache.truncate(new_len)?;
        if let Some(cpu_cache) = self.cpu_cache.lock().unwrap().as_mut() {
            cpu_cache.truncate(new_len)?;
        }
        Ok(())
    }

    pub fn forward_all(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut QuantStreamCache,
    ) -> Result<Tensor> {
        let h = self.forward_hidden(input_ids, index_pos, cache)?;
        let h = self.ln_f.forward(&h)?;
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut QuantStreamCache,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let h = self.forward_hidden(input_ids, index_pos, cache)?;
        let h = self.ln_f.forward(&h)?;
        let h = h.i((.., seq_len - 1, ..))?.contiguous()?;
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }

    /// Hidden-state path: embedding + all transformer blocks across
    /// resident/CPU/streamed tiers, no final norm or lm_head. Returned
    /// shape: `(batch, seq_len, hidden)`. Shared by `forward` (slices to
    /// last position) and `forward_all` (keeps all positions).
    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut QuantStreamCache,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(input_ids)?;

        for (i, block) in self.resident_blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, i, cache)?;
        }

        let cpu_start = self.resident_blocks.len();
        if !self.cpu_resident_blocks.is_empty() {
            let cpu_dev = Device::Cpu;
            let mut x_cpu = x.to_device(&cpu_dev)?;
            let mut cpu_cache_guard = self.cpu_cache.lock().unwrap();
            if cpu_cache_guard.is_none() {
                *cpu_cache_guard = Some(QuantStreamCache::new(
                    cache.use_kv_cache,
                    DType::F32,
                    &self.cfg,
                    &cpu_dev,
                )?);
            }
            let cpu_cache = cpu_cache_guard.as_mut().unwrap();
            for (offset, block) in self.cpu_resident_blocks.iter().enumerate() {
                let i = cpu_start + offset;
                x_cpu = block.forward(&x_cpu, index_pos, i, cpu_cache)?;
            }
            x = x_cpu.to_device(&self.device)?;
        }

        let stream_start = self.resident_blocks.len() + self.cpu_resident_blocks.len();
        let streamed: Vec<std::path::PathBuf> = self.layer_files[stream_start..].to_vec();

        // Optional single-worker prefetcher: while compute runs on layer N,
        // a background thread walks layer N+1 through the OS buffer cache
        // (chunked read, drop data) so the next GgufLayer::open hits warm
        // pages. Off by default because on RAM-tight hosts (e.g. 36 GB Mac
        // running 70B-Q4) the page cache can't shed cold layers fast enough
        // and the prefetch racing ahead can OOM-kill the process. Enable
        // explicitly with `RUSTYLLM_PREFETCH=1` on hosts with headroom.
        let worker_handle = if std::env::var("RUSTYLLM_PREFETCH").as_deref() == Ok("1") {
            let (tx, rx) = std::sync::mpsc::sync_channel::<std::path::PathBuf>(1);
            let worker = std::thread::spawn(move || {
                use std::io::Read;
                let mut buf = vec![0u8; 8 * 1024 * 1024];
                while let Ok(path) = rx.recv() {
                    if let Ok(mut f) = std::fs::File::open(&path) {
                        loop {
                            match f.read(&mut buf) {
                                Ok(0) | Err(_) => break,
                                Ok(_) => {}
                            }
                        }
                    }
                }
            });
            if let Some(first) = streamed.first() {
                let _ = tx.send(first.clone());
            }
            Some((tx, worker))
        } else {
            None
        };

        for (offset, path) in streamed.iter().enumerate() {
            let i = stream_start + offset;

            if let Some((tx, _)) = worker_handle.as_ref() {
                if offset + 1 < streamed.len() {
                    let _ = tx.send(streamed[offset + 1].clone());
                }
            }

            let mut layer = GgufLayer::open(path, &self.device)?;
            let block = QBlock::load(&mut layer, i, &self.cfg)?;
            // Block now owns the tensor data; drop the file buffer so
            // peak RSS doesn't grow by ~500 MB per layer across the loop.
            layer.release_buffer();
            x = block.forward(&x, index_pos, i, cache)?;
        }

        if let Some((tx, worker)) = worker_handle {
            drop(tx);
            let _ = worker.join();
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::{quantize_shard, Quantization};
    use std::collections::HashMap as Map;
    use tempfile::tempdir;

    fn random_weight(rows: usize, cols: usize) -> Tensor {
        Tensor::randn(0f32, 1f32, (rows, cols), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    /// Smoke test: quantize a synthetic single-layer shard, reload it
    /// through the streaming reader, and run a tiny forward pass. Uses
    /// dims small enough that Q4_0's 32-element block size divides
    /// cleanly. Verifies the GGUF -> QMatMul -> forward path works
    /// end-to-end against a synthetic Llama-like layer.
    #[test]
    fn quantized_block_loads_and_runs() {
        let cfg = Config {
            hidden_size: 64,
            intermediate_size: 64,
            vocab_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            max_position_embeddings: 32,
            rope_scaling: None,
            use_flash_attn: false,
        };

        let dir = tempdir().unwrap();
        let f16_path = dir.path().join("model.layers.0.safetensors");
        let gguf_path = dir.path().join("model.layers.0.gguf");

        // Build synthetic layer weights.
        let mut t: Map<String, Tensor> = Map::new();
        let p = "model.layers.0";
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            t.insert(
                format!("{p}.self_attn.{proj}.weight"),
                random_weight(cfg.hidden_size, cfg.hidden_size),
            );
        }
        for proj in ["gate_proj", "up_proj"] {
            t.insert(
                format!("{p}.mlp.{proj}.weight"),
                random_weight(cfg.intermediate_size, cfg.hidden_size),
            );
        }
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            random_weight(cfg.hidden_size, cfg.intermediate_size),
        );
        t.insert(
            format!("{p}.input_layernorm.weight"),
            Tensor::ones((cfg.hidden_size,), DType::F16, &Device::Cpu).unwrap(),
        );
        t.insert(
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::ones((cfg.hidden_size,), DType::F16, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&t, &f16_path).unwrap();

        quantize_shard(&f16_path, &gguf_path, Quantization::Q4_0).unwrap();

        // Build a single quantized block and run it on a random hidden state.
        let mut layer = GgufLayer::open(&gguf_path, &Device::Cpu).unwrap();
        let block = QBlock::load(&mut layer, 0, &cfg).unwrap();
        let mut cache = QuantStreamCache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 4, cfg.hidden_size), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let y = block.forward(&x, 0, 0, &mut cache).unwrap();
        assert_eq!(y.dims(), &[1, 4, cfg.hidden_size]);
    }
}
