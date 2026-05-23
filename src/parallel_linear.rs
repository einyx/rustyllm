//! Pure-Rust parallel Linear shim — rayon-parallel matmul for the
//! batch-size-1 inference shapes used by the streaming Block, without
//! requiring Intel MKL or any other BLAS.
//!
//! Why: candle 0.10 without `--features mkl` falls back to a single-
//! threaded matmul for the [1, hidden] @ [hidden, out] shapes we hit
//! every generated token, leaving 31 of 32 cores idle on this host
//! and creating a 50x wall-time gap vs Ollama. Plugging this Linear
//! into the streaming Block in place of `candle_nn::Linear` parallelises
//! over output columns and gets us close to one matmul-per-core.
//!
//! Zero-copy: the weight tensor is held as a candle `Tensor` and read
//! directly from its `Storage` at forward time. No heap duplication on
//! top of the mmap-backed page cache, so adding this Linear to the
//! streaming Block does NOT increase resident memory per layer.

use std::sync::Arc;

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Bulk-convert one row of f16 weights to f32, then do a pure-f32
/// dot product. On x86_64 with avx2+fma+f16c (target-feature gate
/// below) we use explicit intrinsics for the dot product to guarantee
/// 8 fused-multiply-adds per cycle — the compiler's auto-vectoriser
/// sometimes drops to a single FMA per loop iteration for this
/// pattern. On other archs / older CPUs we fall back to the scalar
/// loop which the compiler still vectorises well.
#[inline(always)]
fn dot_f16_f32(w_row: &[f16], x_row: &[f32], scratch: &mut [f32]) -> f32 {
    debug_assert_eq!(w_row.len(), x_row.len());
    debug_assert_eq!(scratch.len(), w_row.len());
    // F16C-accelerated bulk conversion (half crate dispatches to
    // _mm256_cvtph_ps when target-cpu supports it).
    w_row.convert_to_f32_slice(scratch);

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        return unsafe { dot_f32_avx2_fma(x_row, scratch) };
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for k in 0..x_row.len() {
            sum += x_row[k] * scratch[k];
        }
        sum
    }
}

/// Manual AVX2+FMA dot product on two f32 slices of equal length.
/// Eight FMAs per inner-loop iteration on aligned 8-wide lanes; tail
/// handled scalar. Safety: caller guarantees `a.len() == b.len()` and
/// the slices are valid f32 memory (checked at call site by
/// debug_assert_eq above).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let av = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    // Horizontal sum of the 8 lanes.
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    let mut sum = buf.iter().sum::<f32>();

    // Scalar tail.
    for i in tail_start..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Drop-in replacement for `candle_nn::Linear` whose forward uses a
/// rayon-parallel F16-weight × F32-activation matmul instead of
/// candle's single-threaded CPU kernel.
pub struct ParallelLinear {
    /// Weight `[out_features, in_features]`, held as candle Tensor so
    /// the underlying storage (e.g. mmap-backed safetensors page) isn't
    /// duplicated to the heap. Required to be CPU+F16 with a contiguous
    /// layout; checked at construction.
    weight: Arc<Tensor>,
    /// Optional bias, dequantised to F32 once at construction (small,
    /// only `out_features` floats).
    bias: Option<Arc<Vec<f32>>>,
    in_features: usize,
    out_features: usize,
}

impl ParallelLinear {
    /// Build from a candle weight tensor. Validates the tensor is CPU,
    /// F16, 2D, and contiguous — those are the assumptions the parallel
    /// kernel relies on for safe direct-storage access. Accepts any
    /// dtype that can be converted to F16 (a copy in that case).
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Msg(format!(
                "ParallelLinear: expected 2D weight, got shape {:?}",
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];

        // Force F16, on CPU, contiguous. If the input already meets
        // these, candle's operators short-circuit and there's no copy.
        let weight = weight
            .to_dtype(DType::F16)?
            .to_device(&Device::Cpu)?
            .contiguous()?;

        // Validate at construction time so forward() can use unchecked
        // storage access without re-validating per call.
        {
            let (storage, layout) = weight.storage_and_layout();
            match &*storage {
                Storage::Cpu(CpuStorage::F16(data)) => {
                    let start = layout.start_offset();
                    let expected = out_features * in_features;
                    if data.len() - start < expected {
                        return Err(Error::Msg(format!(
                            "ParallelLinear: storage too small ({}) for weight ({})",
                            data.len() - start,
                            expected
                        )));
                    }
                }
                _ => {
                    return Err(Error::Msg(
                        "ParallelLinear: weight storage is not CPU F16 after coercion".into(),
                    ));
                }
            }
        }

        let bias_f32 = bias
            .map(|b| -> Result<Vec<f32>> {
                let f = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                if f.len() != out_features {
                    return Err(Error::Msg(format!(
                        "ParallelLinear: bias length {} != {}",
                        f.len(),
                        out_features
                    )));
                }
                Ok(f)
            })
            .transpose()?;

        Ok(Self {
            weight: Arc::new(weight),
            bias: bias_f32.map(Arc::new),
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// y = x @ W.T (+ bias). x can be any rank where the last dim is
    /// `in_features`; output reshapes to the same leading dims with
    /// `out_features` in the last position. Output dtype matches input.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let orig_dims: Vec<usize> = x.dims().to_vec();
        if orig_dims.last().copied().unwrap_or(0) != self.in_features {
            return Err(Error::Msg(format!(
                "ParallelLinear: last dim {} != in_features {}",
                orig_dims.last().copied().unwrap_or(0),
                self.in_features
            )));
        }

        let n: usize = orig_dims.iter().take(orig_dims.len() - 1).product();
        let device = x.device().clone();

        // Copy input to a contiguous Vec<f32> on the host so the
        // parallel kernel has cache-friendly access patterns. Input is
        // tiny ([n, in_features] with n≈1), so this copy is free.
        let x_flat = x.reshape((n, self.in_features))?;
        let x_data = x_flat
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let out_features = self.out_features;
        let in_features = self.in_features;
        let bias = self.bias.clone();

        let mut output = vec![0.0f32; n * out_features];

        // Chunk size: ~4 chunks per worker for steal-balancing. Big
        // matmuls (8192-28672 outputs) land at 64-512; small matmuls
        // (1024 outputs on GQA k/v) shrink to 8-32.
        let total = n * out_features;
        let workers = rayon::current_num_threads().max(1);
        let chunk = ((total / (workers * 4)).max(8)).min(total).max(1);

        // Hold a single read guard for the duration of the parallel
        // kernel — rayon workers share the &[f16] reborrow.
        let (storage, layout) = self.weight.storage_and_layout();
        let weight: &[f16] = match &*storage {
            Storage::Cpu(CpuStorage::F16(data)) => &data[layout.start_offset()..],
            _ => {
                return Err(Error::Msg(
                    "ParallelLinear::forward: weight storage drifted from CPU F16".into(),
                ))
            }
        };

        if n == 1 {
            // Decode fast path: with a single input row, x_row is reused
            // across all chunks; we allocate a per-worker scratch buffer
            // for the f16→f32 bulk conversion so the inner dot product
            // becomes pure f32 (vectorisable).
            let x_row = x_data.as_slice();
            output.par_chunks_mut(chunk).enumerate().for_each_init(
                || vec![0.0f32; in_features],
                |scratch, (chunk_idx, slice)| {
                    let start = chunk_idx * chunk;
                    for (offset, out_val) in slice.iter_mut().enumerate() {
                        let j = start + offset;
                        let w_row = &weight[j * in_features..(j + 1) * in_features];
                        let mut sum = dot_f16_f32(w_row, x_row, scratch);
                        if let Some(b) = &bias {
                            sum += b[j];
                        }
                        *out_val = sum;
                    }
                },
            );
        } else {
            output.par_chunks_mut(chunk).enumerate().for_each_init(
                || vec![0.0f32; in_features],
                |scratch, (chunk_idx, slice)| {
                    let start = chunk_idx * chunk;
                    for (offset, out_val) in slice.iter_mut().enumerate() {
                        let pos = start + offset;
                        let b = pos / out_features;
                        let j = pos % out_features;
                        let x_row = &x_data[b * in_features..(b + 1) * in_features];
                        let w_row = &weight[j * in_features..(j + 1) * in_features];
                        let mut sum = dot_f16_f32(w_row, x_row, scratch);
                        if let Some(b) = &bias {
                            sum += b[j];
                        }
                        *out_val = sum;
                    }
                },
            );
        }

        // Drop the storage lock before constructing the output Tensor.
        drop(storage);

        let out = Tensor::from_vec(output, (n, out_features), &device)?;
        let mut shape: Vec<usize> = orig_dims;
        let last = shape.len() - 1;
        shape[last] = out_features;
        let out = out.reshape(shape)?;
        let out = if in_dtype == DType::F32 {
            out
        } else {
            out.to_dtype(in_dtype)?
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Module;
    use candle_nn::Linear;

    #[test]
    fn matches_candle_linear_on_random_input() {
        // Numerical agreement check vs candle's Linear (F32 weight path
        // so candle accepts the dtypes), within F16 round-off.
        let dev = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.1, (128, 256), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let x = Tensor::randn(0f32, 1.0, (3, 256), &dev).unwrap();

        let pl = ParallelLinear::new(weight.clone(), None).unwrap();
        let ours = pl.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        let weight_f32 = weight.to_dtype(DType::F32).unwrap();
        let candle_linear = Linear::new(weight_f32, None);
        let theirs = candle_linear.forward(&x).unwrap().to_vec2::<f32>().unwrap();

        for (row_a, row_b) in ours.iter().zip(theirs.iter()) {
            for (a, b) in row_a.iter().zip(row_b.iter()) {
                let diff: f32 = (a - b).abs();
                assert!(diff < 1e-2, "mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn handles_bias() {
        let dev = Device::Cpu;
        let weight = Tensor::ones((64, 32), DType::F16, &dev).unwrap();
        let bias = Tensor::from_vec(vec![1.0f32; 64], (64,), &dev).unwrap();
        let x = Tensor::ones((1, 32), DType::F32, &dev).unwrap();

        let pl = ParallelLinear::new(weight, Some(bias)).unwrap();
        let y = pl.forward(&x).unwrap();
        let y_vec = y.to_vec2::<f32>().unwrap();
        for v in &y_vec[0] {
            assert!((v - 33.0).abs() < 1e-3);
        }
    }

    #[test]
    fn preserves_input_dtype() {
        let dev = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.1, (16, 32), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let pl = ParallelLinear::new(weight, None).unwrap();

        let x_f16 = Tensor::randn(0f32, 1.0, (1, 32), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let y_f16 = pl.forward(&x_f16).unwrap();
        assert_eq!(y_f16.dtype(), DType::F16);

        let x_f32 = Tensor::randn(0f32, 1.0, (1, 32), &dev).unwrap();
        let y_f32 = pl.forward(&x_f32).unwrap();
        assert_eq!(y_f32.dtype(), DType::F32);
    }

    #[test]
    fn zero_copy_does_not_duplicate_weight() {
        // The point of the zero-copy refactor: ParallelLinear should
        // share the underlying tensor storage. We can't observe that
        // directly, but we can confirm `new()` doesn't error when the
        // input weight is already CPU+F16+contiguous (the no-op path).
        let dev = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.1, (32, 64), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .contiguous()
            .unwrap();
        let pl = ParallelLinear::new(weight, None).unwrap();
        assert_eq!(pl.in_features(), 64);
        assert_eq!(pl.out_features(), 32);
    }
}
