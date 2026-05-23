//! Microbench: ParallelLinear vs candle's Linear on the matmul shapes
//! that actually occur inside one Llama-70B transformer block.
//!
//! Generates random weights, runs each Linear in both implementations,
//! prints per-shape timings and the speedup ratio. Total runtime ~30s;
//! lets us iterate the parallel matmul tuning loop in seconds instead
//! of running the full 16-token bench (~30 min) every time.
//!
//! Usage:
//!   cargo run --release --example bench_parallel_linear

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;
use rustyllm::parallel_linear::ParallelLinear;

/// (label, in_features, out_features) for the Linear shapes inside
/// one Llama-2-70B transformer block. n_heads=64, n_kv_heads=8,
/// hidden=8192, intermediate=28672, head_dim=128.
const LLAMA_70B_SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj  (8192→8192)", 8192, 8192),
    ("k_proj  (8192→1024)", 8192, 1024),
    ("v_proj  (8192→1024)", 8192, 1024),
    ("o_proj  (8192→8192)", 8192, 8192),
    ("gate    (8192→28672)", 8192, 28672),
    ("up      (8192→28672)", 8192, 28672),
    ("down    (28672→8192)", 28672, 8192),
];

fn bench_one(label: &str, in_f: usize, out_f: usize, iters: usize) -> (f64, f64, Tensor, Tensor) {
    let dev = Device::Cpu;
    // F16 weight (what we'd see streamed from a Llama-2-70B shard).
    let w_f16 = Tensor::randn(0f32, 0.02, (out_f, in_f), &dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    // F32 weight for candle's reference (candle's matmul rejects f16×f32).
    let w_f32 = w_f16.to_dtype(DType::F32).unwrap();
    // F32 single-row input (what one transformer block actually sees
    // during per-token streaming decode).
    let x = Tensor::randn(0f32, 1.0, (1, in_f), &dev).unwrap();

    let pl = ParallelLinear::new(w_f16, None).unwrap();
    let candle_linear = Linear::new(w_f32, None);

    // Warm up both implementations (jit caches, alloc, page-in).
    let _ = pl.forward(&x).unwrap();
    let _ = candle_linear.forward(&x).unwrap();

    let t0 = Instant::now();
    let mut p_out = Tensor::zeros(1, DType::F32, &dev).unwrap();
    for _ in 0..iters {
        p_out = pl.forward(&x).unwrap();
    }
    let p_dt = t0.elapsed().as_secs_f64() / iters as f64;

    let t0 = Instant::now();
    let mut c_out = Tensor::zeros(1, DType::F32, &dev).unwrap();
    for _ in 0..iters {
        c_out = candle_linear.forward(&x).unwrap();
    }
    let c_dt = t0.elapsed().as_secs_f64() / iters as f64;

    let _ = label;
    (p_dt, c_dt, p_out, c_out)
}

fn main() {
    println!("ParallelLinear vs candle Linear — Llama-2-70B per-Block shapes");
    println!("rayon workers: {}", rayon::current_num_threads());
    println!();
    println!(
        "{:24} {:>12} {:>12} {:>8}",
        "shape", "parallel(ms)", "candle(ms)", "speedup"
    );
    println!("{}", "-".repeat(60));

    let mut total_p = 0.0;
    let mut total_c = 0.0;
    for (label, in_f, out_f) in LLAMA_70B_SHAPES {
        // Reduce iters for the giant down_proj so total runtime stays
        // bounded (~5s per row, ~30s total).
        let iters = if *in_f * *out_f > 100_000_000 { 20 } else { 50 };
        let (p_ms, c_ms, _, _) = bench_one(label, *in_f, *out_f, iters);
        total_p += p_ms;
        total_c += c_ms;
        println!(
            "{:24} {:12.2} {:12.2} {:7.2}x",
            label,
            p_ms * 1000.0,
            c_ms * 1000.0,
            c_ms / p_ms
        );
    }
    println!("{}", "-".repeat(60));
    println!(
        "{:24} {:12.2} {:12.2} {:7.2}x",
        "(per-block total)",
        total_p * 1000.0,
        total_c * 1000.0,
        total_c / total_p
    );
    println!();
    println!("Per token across all 80 Llama-2-70B layers (matmul only, no load):");
    println!(
        "  parallel: {:.1}s   candle: {:.1}s   {:.1}x faster",
        total_p * 80.0,
        total_c * 80.0,
        total_c / total_p
    );
}
