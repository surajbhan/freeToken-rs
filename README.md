# freeToken-rs

A Rust port of [FreeToken](https://github.com/FlashML-org/FreeToken) — the edge-native
Mixture-of-Experts serving engine from *FreeToken: Efficient Edge-Native MoE Serving with
Bandwidth-Adaptive Execution* ([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)) —
built to measure how the engine's core ideas perform outside the PyTorch stack.

Expert weights live in CPU-cached pinned host RAM; the GPU holds a global LRU cache of
expert slots. Each decode step, routed experts that miss the cache are split between a
batched PCIe fetch and direct CPU compute by the paper's q\* bandwidth ratio, so both
finish together.

## Crates

| crate | what it is |
|---|---|
| `ft-core` | engine core: LRU `SlotCache`, q\* hybrid split policy, copy-run planner, q4_0 quant + AVX2 CPU GEMV (ggml `vec_dot_q4_0_q8_0` port) |
| `ft-gguf` | minimal GGUF v2/v3 reader (header, metadata, mmap'd tensors) |
| `ft-cuda` | CUDA kernels (build-time PTX, `compute_75`+): grouped q4_0 dequant-GEMV with shared-memory activations, silu/reduce, `HostBanks` (`cuMemHostRegister` pinned memory) |
| `ft-bench` | `ft-bench` (bandwidth calibration, port of `ft bench bw`) and `decode` (hybrid MoE decode benchmark, synthetic or real GGUF experts) |

## Results (RTX 4060 Ti, Gemma-4-26B-A4B QAT q4_0, 1862-slot cache)

MoE expert-FFN cost per decode token at ~95% cache-hit routing:

| configuration | ms/token | tok/s ceiling |
|---|---|---|
| naive per-expert kernels | 20.9 | 48 |
| grouped launches (4/layer) | 19.6 | 51 |
| **kernel v2 (smem activations, u16 loads)** | **7.9** | **126** |

Reference: the original Python engine serves the complete model at 14.7 ms/token
(68 tok/s) on the same hardware via its Triton fallback path.

```
cargo test --release          # unit + GPU parity tests
cargo run --release -p ft-bench            # calibrate pcie/cpu bandwidth, q* fraction
cargo run --release -p ft-bench --bin decode -- gguf=model-q4_0.gguf slots=2048 \
    steps=200 warmup=100 fractions=0.0,0.2,1.0 locality=0.9
```

## Notes

- Ported from FreeToken (Apache-2.0); expert-cache semantics, q\* policy, and bank
  layouts follow `python/freetoken/moe/` in the reference implementation.
- Pinned host memory must be allocated with `cuMemHostRegister` on ordinary pages
  (`ft_cuda::HostBanks`), not `CU_MEMHOSTALLOC_WRITECOMBINED` — CPU reads from
  write-combined memory are ~100x slower and the hybrid path reads the same banks.
- In progress: full Gemma-4 forward pass (attention + tokenizer), AVX-512 CPU GEMV,
  OpenAI-compatible serving.
