//! Microbench: ParallelQ4Linear (direct AVX2 Q4_0 × F32) vs candle's
//! QMatMul (Q4_0 via Q8_0 input requantization).

use std::sync::Arc;
use std::time::Instant;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use rustyllm::q4_kernel::ParallelQ4Linear;

const SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj  (8192→8192)", 8192, 8192),
    ("k_proj  (8192→1024)", 8192, 1024),
    ("v_proj  (8192→1024)", 8192, 1024),
    ("o_proj  (8192→8192)", 8192, 8192),
    ("gate    (8192→28672)", 8192, 28672),
    ("up      (8192→28672)", 8192, 28672),
    ("down    (28672→8192)", 28672, 8192),
];

fn bench(in_f: usize, out_f: usize, iters: usize) -> (f64, f64) {
    let dev = Device::Cpu;
    let w_f32 = Tensor::randn(0f32, 0.02, (out_f, in_f), &dev).unwrap();
    let qt = QTensor::quantize(&w_f32, GgmlDType::Q4_0).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, in_f), &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let ours = ParallelQ4Linear::from_qtensor(&qt).unwrap();
    let theirs = QMatMul::from_arc(Arc::new(qt)).unwrap();

    let _ = ours.forward(&x).unwrap();
    let _ = theirs.forward(&x).unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = ours.forward(&x).unwrap();
    }
    let our_dt = t0.elapsed().as_secs_f64() / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = theirs.forward(&x).unwrap();
    }
    let their_dt = t0.elapsed().as_secs_f64() / iters as f64;

    (our_dt, their_dt)
}

fn main() {
    println!("ParallelQ4Linear vs candle QMatMul — Llama-2-70B per-Block shapes, Q4_0");
    println!("rayon workers: {}", rayon::current_num_threads());
    println!();
    println!(
        "{:24} {:>14} {:>14} {:>8}",
        "shape", "ours(ms)", "candle(ms)", "speedup"
    );
    println!("{}", "-".repeat(64));

    let mut total_ours = 0.0;
    let mut total_theirs = 0.0;
    for (label, in_f, out_f) in SHAPES {
        let iters = if *in_f * *out_f > 100_000_000 { 30 } else { 80 };
        let (ours, theirs) = bench(*in_f, *out_f, iters);
        total_ours += ours;
        total_theirs += theirs;
        println!(
            "{:24} {:14.3} {:14.3} {:7.2}x",
            label,
            ours * 1000.0,
            theirs * 1000.0,
            theirs / ours
        );
    }
    println!("{}", "-".repeat(64));
    println!(
        "{:24} {:14.3} {:14.3} {:7.2}x",
        "(per-block total)",
        total_ours * 1000.0,
        total_theirs * 1000.0,
        total_theirs / total_ours
    );
    println!();
    println!("Per token across 80 Llama-2-70B layers (matmul only):");
    println!(
        "  ours:   {:.2}s   candle: {:.2}s   {:.2}x faster",
        total_ours * 80.0,
        total_theirs * 80.0,
        total_theirs / total_ours
    );
}
