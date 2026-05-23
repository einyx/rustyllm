# Contributing to rustyllm

Thanks for your interest in contributing. This document covers the
practical bits: how to set up, what's expected of a PR, and where each
piece of code lives.

## Setup

```bash
git clone https://github.com/einyx/rustyllm
cd rustyllm
cargo build --release           # CPU only
cargo test --release            # 25 unit tests
```

Optional feature builds:

```bash
cargo build --release --features parallel-cpu   # rayon Llama matmul
cargo build --release --features cuda           # requires nvidia-cuda-toolkit
cargo build --release --features metal          # requires macOS 13+
```

## Repo layout

| Path | What's in it |
|---|---|
| `src/streaming.rs` | Llama-family (Llama, Mistral, Qwen2, InternLM2) per-layer streaming |
| `src/streaming_mixtral.rs` | Mixtral MoE block (8 experts, top-K routing) |
| `src/streaming_chatglm.rs` | ChatGLM2/3 block (2D RoPE, combined QKV, SwiGLU) |
| `src/streaming_baichuan.rs` | Baichuan-13B block (W_pack, ALiBi) |
| `src/streaming_quantized.rs` | Per-layer GGUF streaming via candle QMatMul |
| `src/streaming_family.rs` | `from_pretrained` dispatcher (auto-detect family) |
| `src/parallel_linear.rs` | Zero-copy rayon-parallel F16 matmul shim |
| `src/q4_kernel.rs` | Direct AVX2 Q4_0 dequant + F32 fma (experimental) |
| `src/quantize.rs` | F16 safetensors → Q4_0/Q4_K/Q8_0 GGUF |
| `src/shard.rs` | Split HF checkpoints into per-layer files |
| `src/family.rs` | Architecture detection from `config.json` |
| `src/inference.rs` | `LayeredLlama` (all-resident) + `LoadOptions` |
| `examples/bench_*.rs` | Bench scripts for streaming + per-kernel |
| `.github/workflows/test.yml` | CI: build, test, fmt across feature combos |

## What a good PR looks like

- **One thing per PR.** Small + focused beats big + comprehensive.
- **Tests for any new behavior.** Unit tests live next to the code in
  `#[cfg(test)] mod tests`. Integration-style smoke tests that need
  real model weights are deferred — keep unit tests synthetic.
- **`cargo fmt`** before pushing. CI checks `--check` and will fail PRs
  on whitespace.
- **Both build paths.** Run `cargo test --release` AND
  `cargo test --release --features parallel-cpu`. Feature-gated regressions
  are easy to miss locally.
- **Commit messages with the why.** The "what" is in the diff; the
  commit should explain motivation, especially for performance work
  (include before/after numbers from a microbench).

## Adding a new model family

The shape to follow is `src/streaming_mixtral.rs` (substantial
divergence from Llama) or `src/streaming_baichuan.rs` (combined QKV +
custom attention).

1. Add a `*Config` struct deserialised from the family's `config.json`.
2. Inline the family's Block (attention + MLP + norms) from candle's
   reference if available, or port from HF transformers if not.
3. Wire into `src/family.rs::detect()` so `StreamingFamily` can route
   `from_pretrained` automatically (if the family is Llama-compatible)
   or surface a helpful error pointing at the concrete struct (if it
   needs its own block).
4. If layer-name prefixes diverge from `model.layers.N` (e.g. ChatGLM
   uses `transformer.encoder.layers.N`), add a `default_layer_names()`
   helper and dispatch it via `ShardConfig::for_family`.

## Performance work

Use `examples/bench_parallel_linear.rs` (30s) for matmul tuning and
`examples/bench_q4_kernel.rs` for Q4-specific work. Full end-to-end
benches (`bench_streaming_70b{,_q4}`) need real weights and are slow
(~25 min for 16 tokens on CPU).

When proposing a matmul optimisation, include:
- Microbench numbers before/after across the 7 Llama-70B per-block shapes
- Memory footprint impact (does it add resident allocations per layer?)
- Whether it benefits Q4 path, F16 path, or both

## License

Apache-2.0. By contributing you agree your contributions are licensed
under the same terms (no CLA).
