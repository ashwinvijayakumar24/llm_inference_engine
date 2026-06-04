# PRD — From-Scratch LLM Inference Engine

**Status:** Build roadmap (not started)
**Dev hardware:** Apple M4 MacBook Air (Phases 0–3, correctness + Python engine)
**Kernel & benchmark hardware:** NVIDIA GPUs on GT PACE Phoenix — A100 (dev), H100/H200 (final benchmarks)
**PACE account:** `paceship-simpliearn` — 1000 SUs available, confirmed active
**Implementation:** Hybrid — Python orchestration + CUDA C++ kernels
**Budget:** 10–20 hrs/week, 4 weeks core, extendable to 6–7 for differentiators

## 1. Overview & one-line pitch

**Pitch:** A from-scratch LLM inference engine that loads real Llama-family weights and serves efficient text generation, implementing the memory, batching, and kernel optimizations that production systems (vLLM, llama.cpp) use — built to demonstrate both transformer-internals understanding and serious systems engineering.

This is an **inference and systems** project. The model is fixed and pre-trained; all the engineering value is in *how* tokens get generated — the forward pass, the KV cache, how concurrent requests share memory, and how the hot path is made fast.

The line you want an interviewer to draw: *anyone can call `model.generate()`; you implemented what `generate()` does, plus the serving optimizations around it.*

---

## 2. Goals & non-goals

**Goals**

- Implement the full transformer forward pass yourself (component logic, not library calls)
- Build the inference-specific systems: KV cache, batching, paged memory, quantization, a custom kernel
- Produce defensible, quantified benchmarks vs HuggingFace transformers and llama.cpp
- Every design decision explainable cold in an interview

**Non-goals (explicit — do not build)**

- No training or fine-tuning
- One model family, done well — no zoo of models
- No multi-GPU / distributed inference (list as future work)
- No production cruft: auth, billing, dashboards, rate limiting
- No from-scratch tokenizer unless everything else is done

**The "from-scratch" boundary (important for interview defensibility)**
You will get asked "what does from-scratch mean here?" Have this answer ready and enforce it in code:

- **Library is fine:** array storage (NumPy/torch tensors), the underlying GEMM (cuBLAS), the tokenizer, safetensors parsing.
- **Yours:** every component's logic (RoPE, GQA, RMSNorm, SwiGLU, attention, the residual wiring), the KV cache, the scheduler/batching, the paged memory manager, the quantization path, and the custom attention kernel.

That boundary is exactly the thing that's interview-credible: you're not reimplementing BLAS, you're implementing the model and the serving system.

---

## 3. Target model & why

**Primary: Llama 3.2 1B Instruct.**

- Small enough to iterate fast on the Mac during Phases 0–3 — fp16 weights are roughly 2–2.5 GB, trivial in system RAM. On PACE A100 (40 GB HBM2), fits with enormous headroom for batching and KV cache experiments.
- Modern architecture with exactly the components in scope: RoPE, grouped-query attention (GQA), RMSNorm, SwiGLU FFN, tied/untied output projection.
- Well-documented; `config.json` gives you every hyperparameter.

**First task before any code:** open `config.json` and write down `hidden_size`, `num_attention_heads`, `num_key_value_heads` (the GQA group ratio is `num_attention_heads / num_key_value_heads`), `num_hidden_layers`, `intermediate_size`, `rope_theta`, `rms_norm_eps`, `vocab_size`, and whether embeddings are tied. Do **not** hardcode these from memory — read them and assert against the loaded tensor shapes. Half the "my logits don't match" bugs come from a wrong assumption here.

**Frictionless alternative: Qwen2.5-0.5B or 1.5B.** Near-identical architecture (RoPE, GQA, RMSNorm, SwiGLU), Apache-2.0 and ungated on Hugging Face, so no license-acceptance step. Llama 3.2 is gated (requires accepting Meta's license). If the gating slows you down on day one, start on Qwen2.5 — the engine is the same either way. Pick **one** and commit.

---

## 4. Tech stack

### 4a. CUDA / PACE (primary — kernel work and all benchmarks)

| Layer | Choice | Notes |
| --- | --- | --- |
| Orchestration | Python 3.11+ | wiring, sampling, server, benchmarks |
| Reference math | NumPy | correctness-first forward pass |
| Hot-path kernels | CUDA C++ | `.cu` files compiled with `nvcc` |
| Python↔CUDA binding | nanobind (or pybind11) | bind compiled kernel to Python engine |
| GEMM | cuBLAS | call from CUDA layer; don't hand-roll GEMM |
| Profiling | `nsight-compute`, `nvidia-smi` | identify bottlenecks before optimizing |
| Weights | `safetensors` lib, memory-mapped | lazy load to GPU via `.to(device)` |
| Tokenizer | HF `tokenizers` (Rust-backed) | library is fine, per spec |
| HTTP | FastAPI + uvicorn | OpenAI-compatible `/v1/chat/completions` |
| Build | CMake + nvcc + nanobind | standard CUDA project layout |
| Environment | PACE Phoenix via Slurm | `interactive-cpu2` for dev, `gpu-a100`/`gpu-h100` for benchmarks |

**Dev GPU: A100** (partition `interactive-cpu2`, 40 GB HBM2). The industry reference for CUDA development — every CUDA tutorial, Flash Attention paper, and vLLM benchmark targets Ampere. Start here.

**Benchmark GPU: H100 or H200** (partitions `gpu-h100`, `gpu-h200`). Final benchmark runs only. These are the numbers that go on the resume and in the README.

**PACE workflow for daily kernel dev:**

```bash
# Get an interactive A100 session (do this when writing/testing CUDA code)
salloc --partition=interactive-cpu2 --gres=gpu:a100:1 \
       --account=paceship-simpliearn --time=4:00:00

# Confirm GPU attached
nvidia-smi

# Load CUDA toolkit via modules
module load cuda/12.x
```

**Why cuBLAS for GEMM:** reimplementing a fast GEMM burns a week for no differentiation. Spend the kernel budget on the attention pattern specifically — that's what's interesting, interview-relevant, and unique to your project.

### 4b. Apple M4 (dev tool — Phases 0–3 only)

The Mac is where you develop and validate the Python engine before touching CUDA. All correctness work (forward pass, KV cache, sampling, HTTP server) runs fine on CPU and requires no Slurm queue.

| Layer | Choice | Notes |
| --- | --- | --- |
| Reference math | NumPy | same code that runs on PACE |
| Baseline comparisons | HF transformers (CPU/MPS) | oracle for logit matching |
| Quick iteration | no queue, instant feedback | ideal for Phase 0–3 |

Once Phase 3 is done (Python engine correct and benchmarked), move to PACE for all CUDA kernel work. The Python code runs unchanged on both machines — only the kernel layer is GPU-specific.

---

## 5. System architecture

**Data flow (single request):**

```
prompt
  → Tokenizer (HF lib)
  → Scheduler            ← batches concurrent requests, owns the run loop
  → Model Runner (forward pass):
        embed lookup
        for each of N layers:
            RMSNorm
            GQA attention   ← RoPE applied to q,k; read/write KV cache
            + residual
            RMSNorm
            SwiGLU FFN
            + residual
        final RMSNorm
        LM head (output projection → logits)
  → Sampler (greedy / temp / top-k / top-p)
  → Detokenizer
  → stream token out, loop until EOS / max_tokens
```

**Cross-cutting components:**

- **KV Cache Manager** — sits beside the scheduler. Naive version: one contiguous buffer per sequence. Paged version: fixed-size blocks + a per-sequence block table (the PagedAttention idea).
- **Scheduler** — owns the generation loop. Single-request first; later admits/evicts requests each step for continuous batching.

**Module layout (suggested):**

```
engine/
  loader.py        # safetensors → named tensors, shape asserts vs config
  model.py         # forward pass wiring (calls components)
  components.py    # rope, rmsnorm, gqa, swiglu  (NumPy reference)
  cache.py         # KV cache: naive → paged
  scheduler.py     # run loop, batching
  sampler.py       # greedy/temp/top-k/top-p
  server.py        # FastAPI OpenAI-compatible endpoint
  cli.py
  bench/           # harness + baselines
kernels/
  attention.cu     # CUDA attention kernel (decode hot path)
  bindings.cpp     # nanobind
```

---

## 6. Feature breakdown

### Core (must implement — non-negotiable)

- **Weight loading** — parse safetensors, map names to layers, assert shapes against `config.json`. Memory-mapped.
- **Forward pass** — embeddings, RoPE, GQA (correct head-group broadcast), RMSNorm (watch eps placement), SwiGLU FFN, output projection. *Spec:* logits must match a HF reference within float tolerance.
- **KV cache** — store per-layer K/V, append per step, feed attention. *Spec:* generation with cache produces identical tokens to without, at lower per-step cost.
- **Sampling** — greedy, temperature, top-k, top-p. *Spec:* deterministic with fixed seed; each mode unit-tested on a known distribution.
- **CLI** — `generate --prompt "..." --max-tokens N --temp ...`, streaming output.
- **HTTP** — minimal OpenAI-compatible `/v1/chat/completions`, streaming (SSE). No auth.
- **Benchmark harness** — see §8.

### Differentiating (ranked by value × feasibility — pick the top 2–3)

1. **Quantization (int8 → int4 weight-only).** Highest resume-value-per-hour. Clean before/after story (memory ↓, quality ~flat). *Spec:* dequantize-on-load or on-the-fly; report memory delta + perplexity delta on a small eval set.
2. **Continuous / dynamic batching.** The strongest "this is a serving system, not a tutorial" signal. Scheduler admits new requests and retires finished ones mid-flight. *Spec:* throughput scales with concurrent request count; correctness identical to serial.
3. **PagedAttention-style block KV cache.** Pairs naturally with batching; eliminates per-sequence over-allocation. *Spec:* block table + allocator; handles mixed/adversarial sequence lengths without fragmentation blowup.
4. **Custom CUDA attention kernel.** Your systems flex and the headline differentiator. Implement the decode attention hot path in CUDA C++, profile with nsight-compute, and show a measurable latency reduction vs the reference. *Spec:* matches reference output within tolerance; profiled and faster than the PyTorch baseline on A100.
5. **Speculative decoding (true stretch).** Draft model proposes, target verifies. *Spec:* identical output distribution to the target alone, with measured speedup. Only if 1–4 are done.

---

## 7. Phased build plan

Sequenced by dependency. **Correctness first, optimize second** — every optimization needs a correct reference to diff against.

### Phase 0 — Setup (~½ week)

- Repo, CMake + nanobind "hello world" (prove C++↔Python round-trips a NumPy array) **before** writing real kernels.
- Download weights, inspect `config.json`, dump tensor names + shapes.
- Tokenizer round-trips text → ids → text.
- **PACE environment setup (do this in parallel):** SSH into Phoenix, load CUDA toolkit via `module load cuda`, compile a trivial CUDA "hello world" kernel, confirm `nvcc` works and `nvidia-smi` shows an A100. Do this before Phase 4 so there are no toolchain surprises when you actually need it.
    
    ```bash
    salloc --partition=interactive-cpu2 --gres=gpu:a100:1 \       --account=paceship-simpliearn --time=1:00:00nvidia-smi   # confirm A100 attachedmodule load cuda/12.xnvcc --version
    ```
    
- **Milestone:** Mac toolchain works (load weights, tokenize); PACE toolchain works (nvcc compiles, GPU visible).

### Phase 1 — Correctness (~1.5 weeks)

- NumPy forward pass: all components, greedy decode.
- Build a HF oracle: load the same model in HF transformers, compare **hidden states layer-by-layer**, then final logits.
- **Milestone:** coherent text from real weights; logits match HF within tolerance. *This is the make-or-break milestone.*

### Phase 2 — Usable engine (~1 week)

- KV cache (naive contiguous).
- Sampling (temp/top-k/top-p).
- CLI + OpenAI-compatible streaming HTTP endpoint.
- **Milestone:** interactive single-request generation over CLI and HTTP.

### Phase 3 — Baseline benchmarks (~½ week)

- Harness measuring all §8 metrics.
- Stand up HF transformers (CUDA) and llama.cpp (CUDA) baselines on PACE A100.
- **Milestone:** first resume numbers — your naive engine characterized against both baselines on identical hardware.

### Phase 4 — Differentiators (the extension, pick by remaining time)

Recommended order: **Quantization → Continuous batching → Paged KV → Custom CUDA kernel → Speculative.**

- The custom CUDA kernel moves up in strategic value now that PACE is confirmed — it's the headline for NVIDIA/frontier-AI roles.
- Each feature lands as its own before/after benchmark delta (that's the bullet).
- Do **2–3 fully** rather than 5 half-built. A finished quantization + batching + CUDA kernel trio is a very strong story.

### Phase 5 — Polish (~½ week)

- README: architecture decisions (the *why*), benchmark tables, throughput-vs-batch-size chart, quantization memory/quality table.
- The README is where depth becomes legible to a recruiter who won't read your code.

**Dependency notes:** Paged KV depends on a working KV cache (P2) and pairs with batching. Custom kernel depends on a correct reference (P1). Quantization is independent — can slot earlier if you want it locked first.

---

## 8. Benchmarking plan

**Metrics to capture:**

- **Throughput** — tokens/sec, split prefill vs decode (they're very different).
- **TTFT** — time to first token.
- **ITL** — inter-token latency, report p50 and p99 (not just mean).
- **Peak GPU memory** — VRAM footprint (weights + KV cache + activations), measured via `nvidia-smi` or `torch.cuda.max_memory_allocated()`.
- **Throughput vs batch size** — the curve; the headline plot for continuous batching.
- **Memory vs context length** — shows the paged-KV win.
- **Quantized vs fp16** — memory delta + quality delta (perplexity on a small fixed text set).

**Capture matrix — measure each stage so every optimization has a delta:**

| Config | tok/s | TTFT | p99 ITL | peak VRAM |
| --- | --- | --- | --- | --- |
| naive (no cache) |  |  |  |  |
| + KV cache |  |  |  |  |
| + continuous batching |  |  |  |  |
| + paged KV |  |  |  |  |
| + quantization |  |  |  |  |
| + custom CUDA kernel |  |  |  |  |
| HF transformers CUDA (baseline) |  |  |  |  |
| llama.cpp CUDA (baseline) |  |  |  |  |

**Hardware split:** run dev benchmarks on A100 (`gpu-a100`); run final resume-number benchmarks on H100 or H200 (`gpu-h100` / `gpu-h200`). Report which GPU each number comes from — that specificity is a strength, not a detail to hide.

**Framing the llama.cpp comparison honestly:** llama.cpp is a mature, heavily optimized C++ project and will be faster than yours in absolute terms. That's expected and fine. Your story is (a) the *relative deltas* your own optimizations produce, and (b) how close you get to a production system from scratch. Stating this plainly is a strength — interviewers respect a candidate who frames benchmarks honestly over one who claims to beat llama.cpp.

---

## 9. Risks & where you'll get stuck

- **Logits don't match HF (you will hit this).** Usual culprits: RoPE `theta`/scaling, RMSNorm eps placement (inside vs outside the sqrt), attention scale factor, weight transpose conventions, tokenizer special/BOS tokens. *Mitigation:* diff hidden states layer-by-layer against the HF oracle, not just final output — it localizes the bug instantly.
- **PACE Slurm queue waits.** Interactive A100 sessions can queue during busy periods. *Mitigation:* do all Python/correctness work on the Mac; only move to PACE when you're writing or profiling CUDA code. Don't block Phase 1–3 progress on GPU access.
- **Scratch storage 60-day deletion.** Llama weights are ~2.5 GB — easy to re-download, but annoying. *Mitigation:* store weights in `~/ps-simpliearn-0` (project storage, 1 TB, persistent), not scratch. Use scratch only for active job outputs.
- **CUDA kernel correctness bugs.** GPU race conditions and memory layout errors produce silent wrong outputs. *Mitigation:* always diff your kernel output against the NumPy reference on the same input before benchmarking. Never trust speed numbers from a kernel you haven't validated for correctness first.
- **nvcc / CMake / nanobind build friction.** *Mitigation:* the Phase 0 PACE hello-world task before any real kernel work. Sort out the build system when nothing is at stake.
- **cuBLAS vs hand-rolled GEMM rabbit hole.** Same risk as before. *Mitigation:* use cuBLAS; spend kernel budget on the attention pattern only.
- **Continuous batching concurrency bugs.** *Mitigation:* single-threaded scheduler with an explicit step loop first; prove correctness vs serial; add concurrency only after.
- **Paged KV complexity.** Block table + allocator is fiddly. *Mitigation:* test with adversarial mixed sequence lengths early; assert no leaks across many requests.
- **Scope creep across 5 differentiators.** The biggest risk to the whole project. *Mitigation:* finish 2–3. A polished subset beats five stubs.

---

## 10. Draft resume bullets (earn these — fill placeholders)

Aim to make every number defensible from your benchmark matrix. Note the GPU in every bullet — hardware specificity is a strength.

- "Built an LLM inference engine from scratch (Python + CUDA C++) running Llama 3.2 1B with hand-implemented GQA, RoPE, RMSNorm, and SwiGLU plus a paged KV cache; achieved __ tok/s decode on NVIDIA A100."
- "Wrote a custom CUDA attention kernel for the decode hot path, achieving __% latency reduction vs the PyTorch baseline; profiled with nsight-compute on NVIDIA A100."
- "Implemented continuous batching with a PagedAttention-style block KV cache, improving throughput __× at batch size __ versus naive static batching."
- "Added int4 weight-only quantization, cutting VRAM **% (**→__ GB) with <__ perplexity increase."
- "Benchmarked against HuggingFace transformers and llama.cpp on NVIDIA H100/H200: TTFT __ ms, p99 inter-token latency __ ms, throughput vs batch size 1–__ concurrent requests."

---

### Future work (say this, don't build it)

Multi-GPU/tensor-parallel inference, Flash-Attention-style fused kernels, LoRA adapter hot-swapping, full speculative decoding. Naming these signals you know where the project stops and what's next — which is itself a strong interview signal.