# Resume Bullets

All numbers traceable to CSV rows in `bench/results/`, measured on **NVIDIA A100 40GB** (GT PACE Phoenix). Note the GPU in every bullet — hardware specificity is a strength. Only claims actually earned are listed (no paged KV, continuous batching, H100/H200, or nsight-compute — see "Honest scope" below).

## Primary bullets (pick 2–4)

- **Built an LLM inference engine from scratch (Python + CUDA C++)** running Llama 3.2 1B with hand-implemented grouped-query attention, RoPE (Llama-3 frequency scaling), RMSNorm, and SwiGLU plus a KV cache; validated against HuggingFace `transformers` layer-by-layer to **<1e-3 logit error with exact greedy-token match**, reaching decode throughput within **~6% of HuggingFace** on an NVIDIA A100.

- **Wrote a custom CUDA decode-attention kernel** (split-KV flash-decoding with warp-shuffle reductions) **33× faster than a naive baseline** and **matching PyTorch `scaled_dot_product_attention` (0.99×)** on an NVIDIA A100; validated to **<1e-3 vs reference across 100+ randomized inputs** before any speed measurement.

- **Implemented int8/int4 weight-only quantization**, cutting model weight memory **39%** (int8) for only **+0.14 perplexity** on a fixed eval set.

- **Built an OpenAI-compatible streaming inference server** (FastAPI/SSE) with a KV cache and a reproducible benchmark harness measuring TTFT, decode tok/s, and p50/p99 inter-token latency; characterized the engine against HuggingFace `transformers` and `llama.cpp` (CUDA) on identical A100 hardware.

## One-line version (if space is tight)

- **From-scratch LLM inference engine (Python + CUDA C++)** for Llama 3.2 1B — hand-implemented GQA/RoPE/RMSNorm/SwiGLU validated layer-by-layer vs HuggingFace, a custom CUDA attention kernel matching PyTorch SDPA (33× over naive), and int8 quantization (−39% memory, +0.14 perplexity) on NVIDIA A100.

## Honest scope (do NOT claim these — they're future work)

- No continuous batching, no paged KV cache (deferred — documented as future work).
- Benchmarks on **A100 only** (not H100/H200).
- Kernel timed with **CUDA events**, not Nsight Compute profiling.
- Quantization win is **memory, not speed** (on-the-fly dequant; a fused low-precision GEMM would recover throughput).

## Interview-prep notes

- **Explain cold:** eps inside the RMSNorm sqrt; GQA broadcast direction (KV heads repeated across query heads); RoPE half-rotation layout + Llama-3 frequency scaling; streaming (online) softmax; the flash-attention combine rule; per-output-channel quantization scales.
- **Honest framing wins:** llama.cpp is ~5× faster (mature C++) — the story is the *relative deltas of my own optimizations* and matching SDPA from scratch. The CUDA kernel gives only ~4% end-to-end because attention isn't the decode bottleneck (Amdahl) — knowing *where the time goes* is the systems skill.
