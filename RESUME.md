# Resume Bullets

Numbers from `bench/results/` on NVIDIA A100 40GB (GT PACE Phoenix). Note the GPU in each bullet — hardware specificity is a strength.

## Tightest version (pick 2–3)

- **Built an LLM inference engine from scratch (Python + CUDA C++)** running Llama 3.2 1B with hand-implemented grouped-query attention, RoPE, RMSNorm, and SwiGLU, validated against HuggingFace `transformers` layer-by-layer to <1e-3 logit error with exact greedy-token match.

- **Wrote a custom CUDA decode-attention kernel** (split-KV flash-decoding with warp-shuffle reductions) that is **33× faster than a naive baseline** and **matches PyTorch's `scaled_dot_product_attention` (0.99×)** on an NVIDIA A100; validated to <1e-3 vs reference across 100+ random inputs and profiled with Nsight.

- **Implemented int8/int4 weight-only quantization**, cutting model weight memory **39%** (int8) with only **+0.14 perplexity** on a fixed eval set.

- **Built an OpenAI-compatible streaming inference server** (FastAPI/SSE) with a KV cache, continuous generation loop, and a reproducible benchmark harness measuring TTFT, decode throughput, and p50/p99 inter-token latency; characterized the engine against HuggingFace `transformers` and `llama.cpp` on identical hardware.

## Notes for interviews

- **Be ready to explain, cold:** why eps goes inside the RMSNorm sqrt; GQA broadcast direction (KV heads repeated across query heads); RoPE half-rotation layout + Llama-3 frequency scaling; streaming (online) softmax; the flash-attention combine rule; per-output-channel quantization scales.
- **Honest framing wins:** llama.cpp is ~5× faster (mature C++) — the story is the *relative deltas of my own optimizations* and matching SDPA from scratch. The CUDA kernel gives only +4% end-to-end because attention isn't the decode bottleneck (Amdahl) — knowing *where the time goes* is the systems skill.
- **Every number is traceable** to a CSV row in `bench/results/`.
