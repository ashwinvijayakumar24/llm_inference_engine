# TASKS.md — LLM Inference Engine Build Plan

Source of truth: `PRD.md`. Tasks sequenced by dependency. Correctness before optimization. Phases 0–3 on Mac (Python/NumPy). Phase 4 on PACE Phoenix (NVIDIA A100 via Slurm). Phase 5 polish.

Legend:
- **Machine:** Mac | PACE
- ⚠️ = high-risk task (wrong implementation hard to catch later — needs extra validation)

---

## Phase 0 — Setup (~½ week)

### Task 0.1 — Repo + Python toolchain
**Description:** Initialize repo structure matching PRD §5 module layout; set up Python environment.
**Acceptance:** `engine/` and `kernels/` directories exist; `pip install -e .` succeeds; `pytest` runs (even if no tests yet); pre-commit/lint config present.
**Dependencies:** none
**Machine:** Mac

Subtasks:
- 0.1.1 Create directory tree (`engine/`, `kernels/`, `tests/`, `bench/`) matching PRD §5. Pass: `tree -L 2` shows expected layout.
- 0.1.2 Create `pyproject.toml` with deps: `numpy`, `safetensors`, `tokenizers`, `transformers` (for oracle), `torch` (CPU/MPS for oracle), `fastapi`, `uvicorn`, `pytest`. Pass: `pip install -e .` exits 0.
- 0.1.3 Initialize git, `.gitignore` (ignore weights, `__pycache__`, `build/`, `*.so`). Pass: `git status` clean after ignoring weight dir.
- 0.1.4 Add empty stub modules (`loader.py`, `model.py`, `components.py`, `cache.py`, `scheduler.py`, `sampler.py`, `server.py`, `cli.py`) — each importable. Pass: `python -c "import engine.model"` exits 0 for each.

### Task 0.2 — Weight download + inspection
**Description:** Pull Llama 3.2 1B Instruct (or Qwen2.5-0.5B fallback) weights; record config + tensor shapes.
**Acceptance:** `config.json` values written into a tracked notes file; tensor name/shape dump committed.
**Dependencies:** 0.1
**Machine:** Mac
⚠️ — wrong hyperparams here propagate silently into every later phase.

Subtasks:
- 0.2.1 Model committed: **Llama 3.2 1B Instruct** (HF gate access granted). Record in `MODEL.md`. Pass: file exists with model ID + HF repo URL.
- 0.2.2 Download weights via `huggingface_hub` snapshot to `weights/` (gitignored). Pass: `weights/model.safetensors` exists, size matches HF page.
- 0.2.3 Write `scripts/inspect_weights.py` — opens safetensors, prints every tensor name + shape + dtype. Pass: output committed to `notes/tensor_dump.txt`.
- 0.2.4 ⚠️ Read `config.json` and record into `notes/model_config.md`: `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, `intermediate_size`, `rope_theta`, `rms_norm_eps`, `vocab_size`, `tie_word_embeddings`, max_position_embeddings. Pass: values match a manual cross-check against HF model card.
- 0.2.5 Compute GQA group ratio = `num_attention_heads / num_key_value_heads`; write derived expected shapes for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Pass: derived shapes match dumped tensor shapes exactly (assert script exits 0).

### Task 0.3 — Tokenizer round-trip
**Description:** Load HF tokenizer for chosen model; verify encode→decode identity.
**Acceptance:** Known prompt round-trips losslessly; special tokens (BOS/EOS/pad) identified.
**Dependencies:** 0.2
**Machine:** Mac

Subtasks:
- 0.3.1 Write `scripts/tokenizer_check.py`: load tokenizer, encode "Hello, world!", decode, assert equal. Pass: script exits 0.
- 0.3.2 Print and record BOS, EOS, pad token IDs in `notes/model_config.md`. Pass: values match `tokenizer_config.json`.
- 0.3.3 Encode chat-formatted prompt (apply chat template). Pass: output token IDs match HF `apply_chat_template` byte-for-byte.

### Task 0.4 — PACE environment bootstrap (parallel track)
**Description:** Verify PACE access, CUDA toolchain, GPU visibility — before any CUDA code is needed.
**Acceptance:** `nvcc` compiles and runs a hello-world kernel on an A100 interactive session.
**Dependencies:** none (run in parallel with 0.1–0.3)
**Machine:** PACE

Subtasks:
- 0.4.1 SSH into Phoenix; confirm `paceship-simpliearn` account active (`sacctmgr show user $USER`). Pass: account listed.
- 0.4.2 `salloc --partition=interactive-cpu2 --gres=gpu:a100:1 --account=paceship-simpliearn --time=1:00:00`; confirm `nvidia-smi` shows A100. Pass: A100 row visible.
- 0.4.3 `module load cuda/12.x`; `nvcc --version` runs. Pass: nvcc reports a 12.x version.
- 0.4.4 Write `kernels/hello.cu` that prints from device; compile and run. Pass: kernel output prints to stdout.
- 0.4.5 Verify project storage `~/ps-simpliearn-0` exists and is writable; mark as weights dir for PACE. Pass: `touch ~/ps-simpliearn-0/.testfile` succeeds.

### Task 0.5 — nanobind round-trip (PACE)
**Description:** Prove C++↔Python NumPy round-trip via nanobind before writing real kernels.
**Acceptance:** A Python script calls a compiled C++ function that receives a NumPy array and returns a modified one.
**Dependencies:** 0.4
**Machine:** PACE

Subtasks:
- 0.5.1 Set up `CMakeLists.txt` with nanobind + CUDA. Pass: `cmake -B build && cmake --build build` succeeds.
- 0.5.2 Write `kernels/bindings.cpp` exposing `add_one(np.ndarray) -> np.ndarray`. Pass: Python test asserts result == input+1.
- 0.5.3 Add CUDA version: `add_one_cuda` that moves array to GPU, runs trivial kernel, returns. Pass: result == input+1 and `nvprof`/`nsys` shows GPU activity.

**Phase 0 milestone:** Mac toolchain loads weights and tokenizes; PACE toolchain compiles CUDA and rounds-trips NumPy through C++.

---

## Phase 1 — Correctness (~1.5 weeks) ⚠️ MAKE-OR-BREAK PHASE

All Phase 1 work runs on Mac. Every component validated against HF transformers oracle. No optimization, no batching, no cache yet.

### Task 1.1 — HF oracle harness
**Description:** Load same model in HF transformers; expose hooks to capture hidden states at every layer.
**Acceptance:** For a fixed prompt + fixed seed, can dump per-layer hidden state tensors to disk for later diffing.
**Dependencies:** 0.2, 0.3
**Machine:** Mac
⚠️ — the oracle IS the ground truth. If oracle capture is wrong, every later "match" is meaningless.

Subtasks:
- 1.1.1 Write `tests/oracle.py`: load model with `AutoModelForCausalLM`, set eval mode, fixed seed, fp32. Pass: forward pass on a known prompt produces deterministic logits across two runs.
- 1.1.2 Register forward hooks on every transformer layer's input and output. Pass: hook fires N_layers times per forward.
- 1.1.3 Hook captures: post-embedding, post-each-attention, post-each-FFN, post-final-norm, final logits. Pickle to `tests/fixtures/oracle_<prompt_hash>.pkl`. Pass: loading pickle yields tensors with expected shapes.
- 1.1.4 Add `compare_tensors(a, b, name, atol, rtol)` helper logging max-abs-diff and arg-max-diff position. Pass: comparing oracle vs itself yields 0 diff.

### Task 1.2 — Weight loader
**Description:** Parse safetensors → name-keyed dict; assert all shapes against config-derived expectations.
**Acceptance:** Every tensor loaded; every shape asserted; missing/extra tensors raise.
**Dependencies:** 0.2
**Machine:** Mac
⚠️ — silent transpose or permutation bugs surface only at logit-mismatch time.

Subtasks:
- 1.2.1 `engine/loader.py`: open safetensors with mmap, return `dict[str, np.ndarray]`. Pass: keys match `notes/tensor_dump.txt`.
- 1.2.2 Build expected-shape table from config (per layer: q/k/v/o, gate/up/down, norms; global: embed, final_norm, lm_head). Pass: every loaded tensor's shape == expected shape; assertion fails loudly otherwise.
- 1.2.3 Detect `tie_word_embeddings`; if true, lm_head reuses embed weights. Pass: assertion that lm_head tensor is absent when tied, present otherwise.
- 1.2.4 Cast to fp32 for reference impl (Mac CPU). Pass: dtype check on returned arrays.

### Task 1.3 — RMSNorm
**Description:** Implement RMSNorm in NumPy.
**Acceptance:** Output matches HF `LlamaRMSNorm` within 1e-5 atol on random input + real weight.
**Dependencies:** 1.1, 1.2
**Machine:** Mac
⚠️ — eps placement (inside vs outside sqrt) is the classic bug.

Subtasks:
- 1.3.1 Implement `rms_norm(x, weight, eps)` in `components.py`. Confirm formula: `x * weight / sqrt(mean(x^2) + eps)`. Pass: unit test on random input matches a hand-computed reference.
- 1.3.2 ⚠️ Diff against HF `LlamaRMSNorm` on random input using one real layer's `input_layernorm.weight`. Pass: max-abs-diff < 1e-5.
- 1.3.3 Test fp32 input both small (~1e-6) and large (~1e3) magnitudes. Pass: relative error < 1e-4 in both.

### Task 1.4 — RoPE
**Description:** Implement rotary position embedding.
**Acceptance:** Applied to q,k matches HF rotary application within 1e-5.
**Dependencies:** 1.3
**Machine:** Mac
⚠️ — `rope_theta`, half-rotation vs interleaved layout, and position indexing are all common silent bugs.

Subtasks:
- 1.4.1 Precompute `cos`/`sin` tables of shape `(max_seq, head_dim)` using `rope_theta` from config. Pass: values match `LlamaRotaryEmbedding` output for positions 0..16.
- 1.4.2 ⚠️ Confirm Llama uses the half-rotation layout (first half / second half) vs interleaved — inspect HF source for chosen model. Document choice in code comment.
- 1.4.3 Implement `apply_rope(q, k, positions)`. Pass: diff vs HF on random q,k at positions 0..32 < 1e-5.
- 1.4.4 Test position offset (prefill at 0, decode at N). Pass: applying at position N matches HF generating after N prior tokens.

### Task 1.5 — GQA attention (no cache yet)
**Description:** Single-step grouped-query attention with KV head broadcasting.
**Acceptance:** Output matches HF attention block on real layer-0 input within 1e-4.
**Dependencies:** 1.4
**Machine:** Mac
⚠️ — GQA broadcast direction (repeat KV across query heads) is the classic GQA bug; scale factor `1/sqrt(head_dim)` placement matters.

Subtasks:
- 1.5.1 Project q (n_heads), k/v (n_kv_heads); reshape to `(seq, n_heads, head_dim)` and `(seq, n_kv_heads, head_dim)`. Pass: shape assertions.
- 1.5.2 Apply RoPE to q, k.
- 1.5.3 ⚠️ Repeat k,v across query groups: `n_heads / n_kv_heads` copies of each KV head. Pass: shape becomes `(seq, n_heads, head_dim)`; specific head-to-kv-head mapping documented and tested.
- 1.5.4 Compute scaled-dot-product with causal mask: `softmax(qk^T / sqrt(head_dim) + mask) v`. Pass: mask blocks future positions (verify by zeroing future attention weights manually).
- 1.5.5 Output projection. Diff vs HF layer-0 attention output on a 16-token prompt. Pass: max-abs-diff < 1e-4.

### Task 1.6 — SwiGLU FFN
**Description:** Implement SwiGLU feedforward.
**Acceptance:** Output matches HF MLP within 1e-4.
**Dependencies:** 1.3
**Machine:** Mac

Subtasks:
- 1.6.1 Implement `swiglu(x, gate_w, up_w, down_w) = down(silu(gate(x)) * up(x))`. Pass: shape sanity.
- 1.6.2 Diff vs HF MLP on layer-0 post-norm input. Pass: max-abs-diff < 1e-4.

### Task 1.7 — Full forward pass wiring
**Description:** Compose embed → N×(norm+attn+residual, norm+ffn+residual) → final_norm → lm_head.
**Acceptance:** Final logits match HF within tolerance on a fixed prompt.
**Dependencies:** 1.2–1.6, 1.1
**Machine:** Mac
⚠️ — residual wiring direction (pre-norm vs post-norm) and final-norm placement matter.

Subtasks:
- 1.7.1 Embedding lookup. Diff vs HF embed output. Pass: exact match (< 1e-6, no float ops).
- 1.7.2 ⚠️ Layer-by-layer diff against oracle (1.1.3 pickle): after layer 0 attention, after layer 0 FFN, after layer 1 ... Pass: each step max-abs-diff < 1e-4. **First mismatch localizes the bug — do NOT skip past a failing layer.**
- 1.7.3 Final RMSNorm + LM head. Pass: argmax of logits matches HF argmax for every position in the prompt.
- 1.7.4 ⚠️ Full logit diff on three different prompts (short, medium, edge-case with special tokens). Pass: max-abs-diff < 1e-3, argmax matches on every position.

### Task 1.8 — Greedy decode
**Description:** Loop forward pass to generate tokens (no KV cache yet — recompute every step).
**Acceptance:** Generated text matches HF `model.generate(do_sample=False)` for the same prompt + max_tokens.
**Dependencies:** 1.7
**Machine:** Mac

Subtasks:
- 1.8.1 Implement loop: append token, re-run full prefill each step (slow but correct). Pass: produces text.
- 1.8.2 ⚠️ Compare 32 generated tokens against HF greedy generation. Pass: token IDs match exactly.

**Phase 1 milestone:** Coherent text from real weights; per-layer hidden states and final logits match HF within tolerance; greedy tokens identical to HF.

---

## Phase 2 — Usable engine (~1 week)

### Task 2.1 — KV cache (naive contiguous)
**Description:** Per-layer K,V buffers; append per decode step; attention reads full cache.
**Acceptance:** Generation with cache produces tokens identical to no-cache version.
**Dependencies:** 1.8
**Machine:** Mac
⚠️ — off-by-one in position indexing or cache writeback corrupts only later tokens, easy to miss.

Subtasks:
- 2.1.1 Allocate `K[layer]`, `V[layer]` with shape `(max_seq, n_kv_heads, head_dim)` per sequence. Pass: shape + dtype asserts.
- 2.1.2 Split forward into `prefill(prompt)` (writes cache positions 0..L-1) and `decode_step(token)` (writes cache position L, reads 0..L). Pass: positions written match expected index per step.
- 2.1.3 ⚠️ Diff generated token IDs cache vs no-cache on three prompts, 64 tokens. Pass: identical sequences.
- 2.1.4 Benchmark per-step latency cache vs no-cache. Pass: cached decode step substantially faster than full re-prefill.

### Task 2.2 — Sampling
**Description:** Greedy, temperature, top-k, top-p; deterministic with seed.
**Acceptance:** Each sampler unit-tested on a known distribution.
**Dependencies:** 2.1
**Machine:** Mac

Subtasks:
- 2.2.1 `greedy(logits) -> argmax`. Pass: unit test.
- 2.2.2 `temperature(logits, T)`. Pass: T=1 identity; T→0 approaches greedy; T→∞ approaches uniform.
- 2.2.3 `top_k(logits, k)`. Pass: only top-k entries non-zero post-filter.
- 2.2.4 `top_p(logits, p)`. Pass: cumulative prob of kept tokens ≥ p, minimal set.
- 2.2.5 Seed RNG; verify identical outputs across two runs with same seed. Pass: bit-exact match.

### Task 2.3 — CLI
**Description:** `generate --prompt --max-tokens --temp --top-k --top-p --seed`, streaming token output.
**Acceptance:** Interactive end-to-end generation from terminal.
**Dependencies:** 2.2
**Machine:** Mac

Subtasks:
- 2.3.1 Argparse interface in `engine/cli.py`. Pass: `--help` shows all flags.
- 2.3.2 Stream tokens to stdout as generated (flush per token). Pass: visible streaming on a slow prompt.
- 2.3.3 Stop on EOS or max_tokens. Pass: stops correctly in both conditions.

### Task 2.4 — HTTP server (OpenAI-compatible)
**Description:** FastAPI `/v1/chat/completions`, SSE streaming, no auth.
**Acceptance:** `curl` with OpenAI-format payload returns streamed completion.
**Dependencies:** 2.3
**Machine:** Mac

Subtasks:
- 2.4.1 FastAPI app skeleton with `/v1/chat/completions`. Pass: server starts, `/docs` renders.
- 2.4.2 Parse `messages[]` → tokens via chat template. Pass: same tokens as 0.3.3.
- 2.4.3 Stream `data: {...}\n\n` SSE chunks matching OpenAI schema. Pass: `curl -N` shows token-by-token stream.
- 2.4.4 Non-stream mode also supported. Pass: `stream=false` returns single JSON.

**Phase 2 milestone:** Interactive single-request generation via CLI and HTTP.

---

## Phase 3 — Baseline benchmarks (~½ week)

### Task 3.1 — Benchmark harness
**Description:** Measure tok/s (prefill, decode), TTFT, p50/p99 ITL, peak VRAM.
**Acceptance:** Reproducible harness produces a CSV row per config.
**Dependencies:** 2.4
**Machine:** Mac (harness dev), PACE (runs)

Subtasks:
- 3.1.1 `bench/harness.py`: prompt set (short/medium/long), N warmup + N measured runs, JSON/CSV output. Pass: dry run produces output file.
- 3.1.2 Per-token timestamps → TTFT, p50/p99 ITL. Pass: synthetic test with known-latency stub returns expected percentiles.
- 3.1.3 Peak VRAM via `torch.cuda.max_memory_allocated()` and `nvidia-smi` polling. Pass: numbers reported and consistent within 5%.
- 3.1.4 Record GPU model + driver + CUDA version in every output row. Pass: row contains hardware metadata.

### Task 3.2 — Port engine to PACE (CUDA path)
**Description:** Get Python engine running on PACE A100 with PyTorch tensors on GPU (still no custom kernels).
**Acceptance:** Same generation, same tokens, on A100.
**Dependencies:** 3.1, 0.4
**Machine:** PACE
⚠️ — dtype + device-placement bugs can corrupt outputs silently.

Subtasks:
- 3.2.1 Replace NumPy with torch tensors on `cuda:0`; preserve all algorithms. Pass: greedy tokens on A100 match Mac NumPy bit-comparable up to fp16/fp32 tolerance.
- 3.2.2 Decide weight dtype (fp16 default for A100). Pass: weights load in fp16; logit-argmax matches HF fp16 reference.
- 3.2.3 Smoke benchmark: 128-token decode. Pass: completes without errors; tok/s recorded.

### Task 3.3 — HF transformers baseline (A100)
**Description:** Run HF `model.generate` on same prompts, same hardware.
**Acceptance:** Baseline row in benchmark CSV.
**Dependencies:** 3.1, 3.2
**Machine:** PACE

Subtasks:
- 3.3.1 Script using HF `pipeline` or raw `generate` with KV cache enabled. Pass: produces tokens.
- 3.3.2 Run harness; record tok/s, TTFT, p99 ITL, VRAM. Pass: CSV row added.

### Task 3.4 — llama.cpp baseline (A100)
**Description:** Build llama.cpp with CUDA; run same prompts.
**Acceptance:** Baseline row in CSV.
**Dependencies:** 3.1
**Machine:** PACE

Subtasks:
- 3.4.1 Clone + build llama.cpp with `LLAMA_CUBLAS=1`. Pass: binary runs `--help`.
- 3.4.2 Convert model to GGUF (or download equivalent quant). Pass: model loads.
- 3.4.3 Run harness via llama.cpp's server or CLI; record metrics. Pass: CSV row added.

**Phase 3 milestone:** Naive engine characterized against HF transformers and llama.cpp on A100.

---

## Phase 4 — Differentiators (PACE only)

All Phase 4 work runs on PACE Phoenix A100. Recommended order: Quantization → Continuous batching → Paged KV → Custom CUDA kernel → (Speculative as stretch). Aim 2–3 finished, not 5 stubs.

### Task 4.1 — Int8/Int4 weight-only quantization
**Description:** Quantize weights post-load; dequant on-the-fly during matmul.
**Acceptance:** Memory drops measurably; perplexity delta small on fixed eval text.
**Dependencies:** 3.3
**Machine:** PACE
⚠️ — quantization correctness is hard to see — wrong scale/zero-point may still produce fluent text but degraded quality.

Subtasks:
- 4.1.1 Implement per-channel int8 symmetric quantization for linear weights (q,k,v,o, gate,up,down). Pass: round-trip dequant of weights within expected error.
- 4.1.2 Replace matmul path: dequant weights on-the-fly OR use packed int8 GEMM. Pass: forward pass runs; logits diff vs fp16 < some bound (document).
- 4.1.3 ⚠️ Perplexity eval on a small fixed text (e.g., wikitext 1k tokens) for fp16 vs int8. Pass: delta < 0.5 perplexity, documented.
- 4.1.4 Extend to int4 group-wise (groups of 32 or 128). Pass: VRAM drops ~4×; perplexity delta documented.
- 4.1.5 Benchmark row added: memory delta + tok/s + perplexity.

### Task 4.2 — Continuous batching
**Description:** Scheduler admits new requests and retires finished ones mid-step.
**Acceptance:** Throughput scales with concurrency; per-request output identical to serial.
**Dependencies:** 2.4
**Machine:** PACE
⚠️ — concurrency bugs can produce subtly wrong outputs for some requests; needs identity tests.

Subtasks:
- 4.2.1 Refactor scheduler: explicit `step()` over a "running" set; each step processes one decode token per active sequence. Pass: serial run matches Phase 2 generation.
- 4.2.2 Admit new requests at step boundaries; retire on EOS/max_tokens. Pass: requests start at staggered times; each finishes correctly.
- 4.2.3 ⚠️ Stress test: 16 concurrent requests with varying prompt lengths; diff each request's output against a serial run. Pass: bit-identical outputs per request.
- 4.2.4 Throughput-vs-batch-size sweep (1, 2, 4, 8, 16, 32). Pass: CSV/plot generated.

### Task 4.3 — Paged KV cache
**Description:** Block-based KV memory + per-sequence block table (PagedAttention-style).
**Acceptance:** Mixed sequence lengths run without fragmentation blowup; correctness identical.
**Dependencies:** 4.2 (pairs with batching)
**Machine:** PACE
⚠️ — block-table off-by-ones and leaked blocks are easy to miss until long-running stress test.

Subtasks:
- 4.3.1 Allocator: fixed block size (e.g., 16 tokens), free-list + alloc/free. Pass: alloc-then-free returns to initial free count after many cycles.
- 4.3.2 Per-sequence block table; attention reads K/V via gather over block table. Pass: output diff vs naive contiguous cache < 1e-5 per token.
- 4.3.3 ⚠️ Adversarial mix: 100 requests with sequence lengths spanning 8–2048; run to completion; assert zero leaked blocks. Pass: free-block count returns to initial.
- 4.3.4 Memory-vs-context-length plot vs naive cache. Pass: paged uses ~block-aligned memory; naive uses max_seq always.

### Task 4.4 — Custom CUDA attention kernel (decode hot path)
**Description:** Implement decode-time attention in CUDA C++; replace PyTorch reference.
**Acceptance:** Bit/tolerance-equivalent output to reference AND measurable latency reduction on A100.
**Dependencies:** 4.3 (or 2.1 at minimum)
**Machine:** PACE
⚠️ — silent wrong outputs from race conditions or layout bugs. Always diff against NumPy/torch reference BEFORE benchmarking.

Subtasks:
- 4.4.1 Spec kernel signature: inputs (Q for one token, K/V cache slices, scale, mask) → output (attention output per head). Pass: signature documented.
- 4.4.2 Implement v1: one CUDA block per (sequence, head); shared-memory tiling over KV. Pass: compiles via nvcc.
- 4.4.3 ⚠️ Correctness diff vs torch reference on 100 random inputs (varying seq lengths). Pass: max-abs-diff < 1e-3 in fp16 across all inputs. **Do not move on if any input fails.**
- 4.4.4 Bind via nanobind; replace decode attention call in engine. Pass: end-to-end greedy tokens identical to pre-kernel engine.
- 4.4.5 Profile with `ncu` (nsight-compute); record occupancy, memory throughput, achieved FLOPs. Pass: profile saved.
- 4.4.6 Iterate: increase tile size, async memory loads, warp-level reductions. Pass: each iteration adds a CSV row with latency delta.
- 4.4.7 Final benchmark vs PyTorch SDPA on A100. Pass: documented latency reduction (target: meaningful — e.g., ≥1.2×).

### Task 4.5 — Speculative decoding (stretch — deferred)
**Description:** Smaller draft model proposes K tokens; target verifies and accepts longest matching prefix. **Stretch goal — does NOT influence primary model selection.** Draft model question deferred until 4.1–4.4 done; do not revisit mid-build.
**Acceptance:** Output distribution identical to target alone; measurable speedup.
**Dependencies:** 4.1–4.4 done
**Machine:** PACE
⚠️ — accept/reject logic must produce same distribution as target sampling; easy to violate.

Subtasks:
- 4.5.1 Pick draft model (deferred decision: smallest viable — Llama 3.2 has no sub-1B sibling, so candidates are external small models or skip). Load both. Pass: both forward passes work.
- 4.5.2 Implement draft-then-verify loop with K=4. Pass: runs end-to-end.
- 4.5.3 ⚠️ Verify output distribution: with greedy, generated tokens must equal target-alone greedy. Pass: 256 tokens identical across both paths.
- 4.5.4 Measure acceptance rate and effective tok/s. Pass: speedup documented; if <1× drop the feature.

### Task 4.6 — Final benchmark sweep (H100/H200)
**Description:** Re-run capture matrix on H100 or H200 partition for resume numbers.
**Acceptance:** All §8 PRD matrix rows filled with H100/H200 numbers, GPU labeled.
**Dependencies:** all chosen Phase 4 differentiators
**Machine:** PACE (`gpu-h100` or `gpu-h200`)

Subtasks:
- 4.6.1 Submit Slurm batch job for H100/H200. Pass: job completes, CSV produced.
- 4.6.2 Each row labeled with GPU SKU + driver + CUDA + commit SHA. Pass: metadata present.
- 4.6.3 Generate throughput-vs-batch-size and memory-vs-context-length plots. Pass: PNGs in `bench/plots/`.

**Phase 4 milestone:** 2–3 differentiators finished with before/after deltas on real hardware.

---

## Phase 5 — Polish (~½ week)

### Task 5.1 — README
**Description:** Architecture, decisions, benchmark tables, plots.
**Acceptance:** A recruiter reading only README understands scope, results, and why.
**Dependencies:** 4.6
**Machine:** Mac

Subtasks:
- 5.1.1 Architecture diagram (data flow per PRD §5). Pass: rendered diagram in README.
- 5.1.2 Component table: what's from-scratch vs library, with justification. Pass: table present.
- 5.1.3 Benchmark capture matrix populated. Pass: matches PRD §8 schema.
- 5.1.4 Throughput-vs-batch-size and memory-vs-context plots embedded. Pass: images render.
- 5.1.5 Quantization memory + quality table. Pass: present.
- 5.1.6 Honest framing of llama.cpp comparison (PRD §8 note). Pass: written.
- 5.1.7 Future work section (PRD §11). Pass: written.

### Task 5.2 — Resume bullets
**Description:** Fill PRD §10 placeholders with actual numbers.
**Acceptance:** Every number traceable to a benchmark CSV row.
**Dependencies:** 5.1
**Machine:** Mac

Subtasks:
- 5.2.1 For each draft bullet, fill in number + GPU. Pass: every number has a CSV row citation in `bench/`.

---

## High-Risk Task Summary (⚠️)

Tasks where silent wrong implementations are easy to ship undetected:

1. **0.2.4** — Config values (RoPE theta, eps, GQA ratio) — propagates everywhere.
2. **1.1** — Oracle harness — ground truth itself.
3. **1.2** — Weight loader — transpose/permutation bugs.
4. **1.3.2** — RMSNorm eps placement.
5. **1.4** — RoPE layout + theta.
6. **1.5.3** — GQA broadcast direction.
7. **1.7.2 / 1.7.4** — Layer-by-layer logit diff (the make-or-break).
8. **1.8.2** — Greedy token-ID match vs HF.
9. **2.1.3** — KV cache identity vs no-cache.
10. **3.2.1** — Mac→PACE dtype/device migration.
11. **4.1.3** — Quantization perplexity eval.
12. **4.2.3** — Continuous batching per-request identity.
13. **4.3.3** — Paged KV adversarial stress + leak check.
14. **4.4.3** — CUDA kernel correctness BEFORE benchmarking.
15. **4.5.3** — Speculative decoding distribution preservation.
