//! Direct AVX2 Q4_0 × F32 matmul, bypassing candle's QMatMul abstraction.
//!
//! Why bother: candle's `QMatMul` for Q4_0 follows the llama.cpp pattern
//! of quantizing the input activation to Q8_0 first, then calling a
//! highly-tuned Q8×Q4 integer dot product. That's the right call for
//! batched workloads where the input requantization cost amortises over
//! many output positions.
//!
//! For per-token streaming inference (batch=1, single-row input), the
//! input has only `in_features` elements (~8 KB for Llama-2-70B) but the
//! requantize-then-dispatch overhead is fixed per matmul. Skipping it
//! and instead dequantizing Q4 weights directly into the F32 FMA loop
//! is measurably faster for these shapes, especially after rayon
//! parallelises across output rows.
//!
//! Block format (ggml Q4_0, 18 bytes for 32 weights):
//!   - bytes 0..2:   f16 scale (one per block)
//!   - bytes 2..18:  16 bytes of packed 4-bit quants, low nibble first
//! Dequant: `weight[i] = (qs[i/2] >> (4*(i%2)) & 0xF - 8) * scale`
//!
//! Total weight footprint: 4.5 bits per parameter (vs 16 bits F16), so
//! per-layer disk reads shrink ~3.6× when streaming.
//!
//! ## Status (not wired into the streaming Block)
//!
//! `examples/bench_q4_kernel.rs` microbenched this against candle's
//! `QMatMul` on Llama-2-70B per-block shapes (32 rayon workers on a
//! 13900K):
//!
//! ```text
//! shape                          ours(ms)  candle(ms)  speedup
//! q_proj  (8192→8192)               2.140       1.756    0.82x
//! k_proj  (8192→1024)               0.313       0.597    1.91x
//! v_proj  (8192→1024)               0.296       0.589    1.99x
//! o_proj  (8192→8192)               2.234       1.655    0.74x
//! gate    (8192→28672)              7.521       5.387    0.72x
//! up      (8192→28672)              7.420       4.871    0.66x
//! down    (28672→8192)              7.341       5.681    0.77x
//! ```
//!
//! candle wins ~33% on the wide shapes that dominate Llama-70B: its
//! Q8-requantize-then-Q8×Q4 integer dot path uses `vpmaddubsw` to pack
//! 32 multiply-adds per AVX2 instruction; our F32 FMA path only gets
//! 8 per `vfmadd231ps`. AVX-512 VNNI would close the gap but Raptor
//! Lake P-cores don't expose it.
//!
//! We do win ~2× on the narrow GQA k/v_proj shapes (1024 out) because
//! candle's per-call Q8 requantization is fixed cost and amortises
//! poorly when the output is small. A hybrid (this kernel for narrow,
//! QMatMul for wide) is plausible but the absolute win is small —
//! matmul is ~4% of per-token time in the streaming Q4 path.
//!
//! Net: kept as a reference implementation + benchmark target. The
//! streaming hot path stays on candle's QMatMul until/unless we find
//! a higher-leverage angle (async H2D, smaller model, etc.).

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use half::f16;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Number of weights packed per Q4_0 block.
pub const QK4_0: usize = 32;
/// Bytes per Q4_0 block (2-byte scale + 16-byte packed quants).
pub const BLOCK_SIZE: usize = 18;

/// Q4_0-weight × F32-activation matmul shim, drop-in compatible with
/// `ParallelLinear` (same `forward(&Tensor) -> Result<Tensor>` shape).
///
/// Holds the weight tensor as a `QTensor` so candle still manages the
/// storage (no heap duplication of the 4.5 bits/param blob). At forward
/// time we read the underlying block bytes and run our own AVX2 dequant
/// + FMA loop in parallel across output rows.
pub struct ParallelQ4Linear {
    weight_blocks: Arc<Vec<u8>>,
    in_features: usize,
    out_features: usize,
    blocks_per_row: usize,
}

impl ParallelQ4Linear {
    /// Build from a Q4_0 QTensor. Pulls the block bytes out into an
    /// owned `Vec<u8>` once at construction (small relative to F16
    /// weights — ~140 MB for a Llama-2-70B layer's Q4_0 weights).
    pub fn from_qtensor(qt: &QTensor) -> Result<Self> {
        if qt.dtype() != GgmlDType::Q4_0 {
            return Err(Error::Msg(format!(
                "ParallelQ4Linear: expected Q4_0, got {:?}",
                qt.dtype()
            )));
        }
        let dims = qt.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Msg(format!(
                "ParallelQ4Linear: expected 2D weight, got {dims:?}"
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];
        if in_features % QK4_0 != 0 {
            return Err(Error::Msg(format!(
                "ParallelQ4Linear: in_features {in_features} not divisible by {QK4_0}"
            )));
        }
        let blocks_per_row = in_features / QK4_0;
        let total_blocks = out_features * blocks_per_row;

        // Round-trip through candle's f32 dequantize to get the values,
        // then re-pack into block bytes. This is a one-shot cost at
        // load time; the forward path operates on the packed bytes.
        //
        // (Candle's QTensor exposes block-typed accessors only via
        // pub(crate), so we go via the public dequantize path. The
        // alternative — duplicating ggml's packing logic — would be
        // brittle to future block-format changes upstream.)
        let dequantized = qt
            .dequantize(&Device::Cpu)
            .map_err(|e| Error::Msg(format!("dequant for repack: {e}")))?;
        let flat = dequantized
            .flatten_all()
            .map_err(|e| Error::Msg(format!("flatten: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| Error::Msg(format!("to_vec1: {e}")))?;
        if flat.len() != out_features * in_features {
            return Err(Error::Msg(format!(
                "dequant size mismatch: {} vs {}*{}",
                flat.len(),
                out_features,
                in_features
            )));
        }

        let mut blocks: Vec<u8> = Vec::with_capacity(total_blocks * BLOCK_SIZE);
        for row in 0..out_features {
            for block_idx in 0..blocks_per_row {
                let start = row * in_features + block_idx * QK4_0;
                let chunk = &flat[start..start + QK4_0];
                // Per-block: max absolute value defines the scale.
                let mut max_abs = 0.0f32;
                let mut max_signed = 0.0f32;
                for &v in chunk {
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                        max_signed = v;
                    }
                }
                let d = max_signed / -8.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let scale_f16 = f16::from_f32(d);
                blocks.extend_from_slice(&scale_f16.to_le_bytes());
                let mut packed = [0u8; QK4_0 / 2];
                for i in 0..(QK4_0 / 2) {
                    let lo = (chunk[i] * id + 8.5).clamp(0.0, 15.0) as u8;
                    let hi = (chunk[i + QK4_0 / 2] * id + 8.5).clamp(0.0, 15.0) as u8;
                    packed[i] = lo | (hi << 4);
                }
                blocks.extend_from_slice(&packed);
            }
        }

        Ok(Self {
            weight_blocks: Arc::new(blocks),
            in_features,
            out_features,
            blocks_per_row,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Drop-in equivalent of `ParallelLinear::forward`. Input is
    /// reshaped to `[n, in_features]`, output to the original leading
    /// dims with `out_features` last.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let orig_dims: Vec<usize> = x.dims().to_vec();
        if orig_dims.last().copied().unwrap_or(0) != self.in_features {
            return Err(Error::Msg(format!(
                "ParallelQ4Linear: last dim {} != in_features {}",
                orig_dims.last().copied().unwrap_or(0),
                self.in_features
            )));
        }

        let n: usize = orig_dims.iter().take(orig_dims.len() - 1).product();
        let device = x.device().clone();
        let x_flat = x.reshape((n, self.in_features))?;
        let x_data = x_flat
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let out_features = self.out_features;
        let in_features = self.in_features;
        let blocks_per_row = self.blocks_per_row;
        let weight = self.weight_blocks.clone();

        let mut output = vec![0.0f32; n * out_features];
        let workers = rayon::current_num_threads().max(1);
        let total = n * out_features;
        let chunk = ((total / (workers * 4)).max(8)).min(total).max(1);

        if n == 1 {
            let x_row = x_data.as_slice();
            output
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(chunk_idx, slice)| {
                    let start = chunk_idx * chunk;
                    for (offset, out_val) in slice.iter_mut().enumerate() {
                        let j = start + offset;
                        let row_start = j * blocks_per_row * BLOCK_SIZE;
                        let w_row = &weight[row_start..row_start + blocks_per_row * BLOCK_SIZE];
                        *out_val = dot_q4_0_f32(w_row, x_row, blocks_per_row);
                    }
                });
        } else {
            output
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(chunk_idx, slice)| {
                    let start = chunk_idx * chunk;
                    for (offset, out_val) in slice.iter_mut().enumerate() {
                        let pos = start + offset;
                        let b = pos / out_features;
                        let j = pos % out_features;
                        let x_row = &x_data[b * in_features..(b + 1) * in_features];
                        let row_start = j * blocks_per_row * BLOCK_SIZE;
                        let w_row =
                            &weight[row_start..row_start + blocks_per_row * BLOCK_SIZE];
                        *out_val = dot_q4_0_f32(w_row, x_row, blocks_per_row);
                    }
                });
        }

        let out = Tensor::from_vec(output, (n, out_features), &device)?;
        let mut shape: Vec<usize> = orig_dims;
        let last = shape.len() - 1;
        shape[last] = out_features;
        let out = out.reshape(shape)?;
        let out = if in_dtype == candle_core::DType::F32 {
            out
        } else {
            out.to_dtype(in_dtype)?
        };
        Ok(out)
    }
}

/// Inline Q4_0 dequantize + F32 fma. Per call: processes
/// `blocks_per_row` Q4_0 blocks (= 32 weights each) against the
/// corresponding F32 input slice.
#[inline(always)]
fn dot_q4_0_f32(w_row: &[u8], x_row: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        return unsafe { dot_q4_0_f32_avx2(w_row, x_row, blocks_per_row) };
    }
    #[allow(unreachable_code)]
    dot_q4_0_f32_scalar(w_row, x_row, blocks_per_row)
}

/// Scalar reference implementation. Used as the fallback when AVX2
/// isn't available, and as the correctness oracle for the AVX2 kernel.
fn dot_q4_0_f32_scalar(w_row: &[u8], x_row: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for block_idx in 0..blocks_per_row {
        let bo = block_idx * BLOCK_SIZE;
        let scale = f16::from_le_bytes([w_row[bo], w_row[bo + 1]]).to_f32();
        let qs = &w_row[bo + 2..bo + 18];
        let xo = block_idx * QK4_0;
        // First half: low nibbles of qs[0..16] map to weight[0..16].
        // Second half: high nibbles of qs[0..16] map to weight[16..32].
        let mut block_sum = 0.0f32;
        for i in 0..(QK4_0 / 2) {
            let lo = ((qs[i] & 0x0f) as i8 - 8) as f32;
            let hi = (((qs[i] >> 4) & 0x0f) as i8 - 8) as f32;
            block_sum += x_row[xo + i] * lo;
            block_sum += x_row[xo + i + QK4_0 / 2] * hi;
        }
        sum += scale * block_sum;
    }
    sum
}

/// AVX2 inner kernel. Per block: load 16 bytes of packed quants,
/// unpack to 32 signed bytes, sign-extend through i32 to f32, multiply
/// by the f16 block scale (broadcast), then 4×8-wide FMA against the
/// matching F32 input window. Safety: caller guarantees alignment of
/// chunked slices via the layout invariants checked at `from_qtensor`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_f32_avx2(w_row: &[u8], x_row: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let lo_mask = _mm_set1_epi8(0x0f);
    let eight = _mm_set1_epi8(8);

    for block_idx in 0..blocks_per_row {
        let bo = block_idx * BLOCK_SIZE;
        let scale = f16::from_le_bytes([w_row[bo], w_row[bo + 1]]).to_f32();
        let scale_v = _mm256_set1_ps(scale);

        // Load 16 packed bytes (32 quants).
        let qs_ptr = w_row.as_ptr().add(bo + 2) as *const __m128i;
        let qs = _mm_loadu_si128(qs_ptr);

        // Low nibbles → first 16 weights (each weight is one byte, signed).
        let lo_nib = _mm_and_si128(qs, lo_mask);
        let lo_signed = _mm_sub_epi8(lo_nib, eight);
        // High nibbles → next 16 weights.
        let hi_nib = _mm_and_si128(_mm_srli_epi16(qs, 4), lo_mask);
        let hi_signed = _mm_sub_epi8(hi_nib, eight);

        // Sign-extend i8 → i32 (8 lanes per __m256i), then to f32.
        // Split each 16-byte half into two 8-byte halves for cvt to i32.
        let lo_lo_i32 = _mm256_cvtepi8_epi32(lo_signed);
        let lo_hi_i32 = _mm256_cvtepi8_epi32(_mm_srli_si128(lo_signed, 8));
        let hi_lo_i32 = _mm256_cvtepi8_epi32(hi_signed);
        let hi_hi_i32 = _mm256_cvtepi8_epi32(_mm_srli_si128(hi_signed, 8));

        let lo_lo_f32 = _mm256_cvtepi32_ps(lo_lo_i32);
        let lo_hi_f32 = _mm256_cvtepi32_ps(lo_hi_i32);
        let hi_lo_f32 = _mm256_cvtepi32_ps(hi_lo_i32);
        let hi_hi_f32 = _mm256_cvtepi32_ps(hi_hi_i32);

        // Apply per-block scale.
        let w0 = _mm256_mul_ps(lo_lo_f32, scale_v);
        let w1 = _mm256_mul_ps(lo_hi_f32, scale_v);
        let w2 = _mm256_mul_ps(hi_lo_f32, scale_v);
        let w3 = _mm256_mul_ps(hi_hi_f32, scale_v);

        // Matching 32-wide input slice.
        let xo = block_idx * QK4_0;
        let x0 = _mm256_loadu_ps(x_row.as_ptr().add(xo));
        let x1 = _mm256_loadu_ps(x_row.as_ptr().add(xo + 8));
        let x2 = _mm256_loadu_ps(x_row.as_ptr().add(xo + 16));
        let x3 = _mm256_loadu_ps(x_row.as_ptr().add(xo + 24));

        acc = _mm256_fmadd_ps(w0, x0, acc);
        acc = _mm256_fmadd_ps(w1, x1, acc);
        acc = _mm256_fmadd_ps(w2, x2, acc);
        acc = _mm256_fmadd_ps(w3, x3, acc);
    }

    // Horizontal sum.
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    buf.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;

    fn random_q4_0(rows: usize, cols: usize) -> QTensor {
        // cols must be divisible by QK4_0=32 for Q4_0 quantization.
        assert!(cols % QK4_0 == 0);
        let t = Tensor::randn(0f32, 0.1, (rows, cols), &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::Q4_0).unwrap()
    }

    #[test]
    fn matches_candle_qmatmul_on_random_input() {
        let dev = Device::Cpu;
        let qt = random_q4_0(64, 256);
        let x = Tensor::randn(0f32, 1.0, (1, 256), &dev).unwrap();

        let ours = ParallelQ4Linear::from_qtensor(&qt).unwrap();
        let our_out = ours.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        // Candle's QMatMul as reference.
        let qmm = candle_core::quantized::QMatMul::from_arc(Arc::new(qt)).unwrap();
        let their_out = candle_core::Module::forward(&qmm, &x)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        // Both go through Q4_0 quantization round-trip, so they should
        // agree to within Q4 quantization noise (much larger than F16
        // round-off — Q4 has ~16 levels per scale).
        for (row_a, row_b) in our_out.iter().zip(their_out.iter()) {
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                let diff: f32 = (a - b).abs();
                // Q4 noise floor for in_features=256 input is ~0.5 in
                // worst case; widen slightly for headroom.
                assert!(
                    diff < 2.0,
                    "ours={a} theirs={b} diff={diff} — outside Q4 round-off"
                );
            }
        }
    }

    #[test]
    fn scalar_and_avx2_agree() {
        // Build a Q4 weight, run forward via our parallel path (which
        // dispatches to AVX2 when available); compare to scalar.
        let qt = random_q4_0(32, 128);
        let ours = ParallelQ4Linear::from_qtensor(&qt).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 128), &Device::Cpu).unwrap();
        let avx_out = ours.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        // Compute scalar dot directly via the scalar fn.
        let x_vec = x.to_vec2::<f32>().unwrap();
        let x_row = &x_vec[0];
        let weight = ours.weight_blocks.clone();
        let blocks_per_row = ours.blocks_per_row;
        let mut scalar_out = vec![0.0f32; ours.out_features];
        for j in 0..ours.out_features {
            let row_start = j * blocks_per_row * BLOCK_SIZE;
            let w_row = &weight[row_start..row_start + blocks_per_row * BLOCK_SIZE];
            scalar_out[j] = dot_q4_0_f32_scalar(w_row, x_row, blocks_per_row);
        }

        for (a, b) in avx_out[0].iter().zip(scalar_out.iter()) {
            let diff: f32 = (a - b).abs();
            assert!(diff < 1e-4, "avx={a} scalar={b} diff={diff}");
        }
    }
}
