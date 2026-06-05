# implemented.md — Build Log & Learning Reference

This document tracks every component built in the LLM inference engine — what it does, why it was built, and the concepts behind it. Updated after each phase. Intended as a reference for interviews and for understanding the project end-to-end.

---

## Phase 0 — Setup

### 0.1 — Repo + Python Toolchain ✅

**Completed:** 2026-06-04

#### What was built

| Path | Purpose |
|------|---------|
| `engine/` | All Python inference engine code |
| `engine/loader.py` | Weight loading from safetensors files |
| `engine/model.py` | Forward pass wiring for the full transformer |
| `engine/components.py` | From-scratch math primitives (RMSNorm, RoPE, GQA, SwiGLU) |
| `engine/cache.py` | KV cache (naive contiguous first, paged later) |
| `engine/scheduler.py` | Generation loop and request scheduling |
| `engine/sampler.py` | Token sampling strategies |
| `engine/server.py` | FastAPI OpenAI-compatible HTTP endpoint |
| `engine/cli.py` | Command-line interface for generation |
| `kernels/` | CUDA C++ kernel source files (Phase 4) |
| `tests/` | Test suite — unit tests and oracle comparisons |
| `bench/` | Benchmark harness and result CSVs |
| `scripts/` | One-off utility scripts (weight inspection, tokenizer checks) |
| `notes/` | Config values, tensor shape dumps, model hyperparameters |
| `pyproject.toml` | Package definition and dependency list |
| `.gitignore` | Excludes weights, build artifacts, and `__pycache__` |

#### Why this structure

The directory layout mirrors the PRD §5 specification. Each concern is isolated: the math lives in `components.py`, the wiring in `model.py`, the serving in `server.py`. This makes each file independently testable — you can diff a single component against a HuggingFace oracle without running the full engine.

Stub modules (all currently `raise NotImplementedError`) exist so the import graph is valid from day one. Any file that tries to `import engine.model` will succeed — it just won't run until implemented. This prevents phantom import errors from masking real bugs.

#### Key dependencies and why each was chosen

| Package | Role | Why, not an alternative |
|---------|------|------------------------|
| `numpy` | Reference math for all Phase 1–3 components | Explicit array operations make every formula readable and debuggable. No autograd overhead, no framework magic. |
| `safetensors` | Load model weights | Memory-mapped, safe (no pickle exploit surface), standard for modern HF models. |
| `tokenizers` | Tokenize prompts | Rust-backed Hugging Face library — fast, correct, handles Llama's BPE + chat template. Per PRD: tokenizer is a library boundary, not from-scratch. |
| `transformers` | HuggingFace oracle for correctness validation | The reference implementation we diff against in Phase 1. Not used in the engine's hot path. |
| `torch` | GPU tensor operations (Phase 3+) and oracle forward passes | Needed for the HF oracle and for the PACE GPU path. On Mac: CPU/MPS only. |
| `fastapi` + `uvicorn` | OpenAI-compatible HTTP serving | FastAPI gives typed request/response models and automatic `/docs`. Uvicorn is the ASGI server behind it. |
| `huggingface_hub` | Downloading model weights | Standard HF snapshot download; handles sharded safetensors and auth tokens. |
| `pytest` | Test runner | Standard Python test framework. Every acceptance criterion in TASKS.md maps to a test. |

#### Concepts to know

**Why `pip install -e .` (editable mode)?**  
Installs the package as a live symlink into site-packages. Changes to `engine/` take effect immediately without reinstalling. Essential during active development.

**Why stub modules rather than empty files?**  
Stubs with `raise NotImplementedError` give a clear failure mode: if you accidentally call an unimplemented function, you get an explicit error pointing at the function name, not a confusing `AttributeError` or silent wrong behavior.

**What is `pyproject.toml`?**  
The modern Python package spec (PEP 517/518). Replaces `setup.py`. Defines: package name, version, Python version requirement, runtime dependencies, optional dev dependencies, and entry points (the `llm-generate` CLI command). The `[tool.pytest.ini_options]` section tells pytest where to look for tests.

---

## Phase 0.2 — Weight Download + Inspection ✅

**Completed:** 2026-06-04

#### What was built

| Path | Purpose |
|------|---------|
| `MODEL.md` | Records chosen model, HF repo, and gate status |
| `weights/` | Downloaded model files (gitignored — ~2.5 GB) |
| `scripts/inspect_weights.py` | Opens safetensors, dumps every tensor name/shape/dtype, asserts all shapes against config |
| `notes/tensor_dump.txt` | Full tensor name/shape/dtype output (146 tensors, committed) |
| `notes/model_config.md` | All hyperparameters, derived shapes, RoPE scaling details |

#### Key discoveries

**146 tensors total.** 16 layers × 9 tensors/layer (q/k/v/o, gate/up/down, input_norm, post_attn_norm) + embed + final_norm = 146. No `lm_head.weight` — confirmed tied to `model.embed_tokens.weight`.

**All weights are bfloat16.** NumPy does not support bfloat16 — the safetensors `framework="numpy"` backend fails. Use `framework="pt"` (torch) for loading. In Phase 1 reference code, cast to float32 after loading.

**Llama 3 scaled RoPE ⚠️.** This is NOT standard RoPE. The model uses `rope_type="llama3"` with a frequency scaling factor of 32× for low-frequency components. Plain `rope_theta=500000` without the scaling would produce wrong positional encodings. See `notes/model_config.md` for full spec.

#### Architecture numbers to memorize

```
hidden_size:            2048
num_attention_heads:    32    (query heads)
num_key_value_heads:    8     (KV heads — GQA group ratio = 4)
head_dim:               64
num_hidden_layers:      16
intermediate_size:      8192
vocab_size:             128256
rope_theta:             500000.0
rms_norm_eps:           1e-05
tie_word_embeddings:    true
BOS token:              128000
EOS tokens:             128001, 128008, 128009
```

#### Why inspect and assert shapes before writing any model code?

Wrong tensor shapes or names discovered at inference time produce confusing matmul errors. Discovered at weight-load time (with a name→shape assertion table) they pinpoint the exact misconfiguration. The assert script exits 0 only when every expected tensor exists with the correct shape — this is a one-time investment that saves hours of debugging in Phase 1.

#### Why torch backend for safetensors, not numpy?

Llama 3.2 weights are stored as bfloat16 (Brain Float 16), a 16-bit float format developed by Google Brain. NumPy has no bfloat16 dtype. PyTorch supports it natively. For Phase 1 reference math (NumPy forward pass), weights are cast to float32 immediately after loading — bfloat16 is only the storage format, not the compute format for the reference implementation.

---

## Phase 0.3 — Tokenizer Round-Trip ✅

**Completed:** 2026-06-04

#### What was built

| Path | Purpose |
|------|---------|
| `scripts/tokenizer_check.py` | Round-trip test, special token verification, chat template validation |

#### What the tokenizer does

Llama 3.2 uses **BPE (Byte-Pair Encoding)** with a vocab of 128,256 tokens (128,000 base + 256 byte-level fallbacks + special tokens). It is backed by the Rust `tokenizers` library for speed.

The tokenizer has two distinct jobs:
1. **Raw encoding/decoding** — text → token IDs and back
2. **Chat template** — wraps messages in Llama 3's structured turn format with header tokens (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`)

#### Key findings

**Three valid EOS tokens.** The model stops on any of `{128001, 128008, 128009}`. In the engine's generation loop, check for all three — using only `tok.eos_token_id` (128009) misses the other two.

**No pad token.** `tok.pad_token_id` is `None`. Batched padding is handled explicitly in Phase 4 (continuous batching) — no built-in pad token to rely on.

**`apply_chat_template` with `tokenize=True` returns `BatchEncoding`**, not `list[int]`, when using the fast tokenizer. Access via `.input_ids`. The engine uses `tokenize=False` → string → `tok.encode()` to keep the tokenization step explicit and debuggable.

**Vocab size discrepancy:** `tok.vocab_size` reports 128,000 but `config.json` says 128,256. The extra 256 are special tokens added on top of the base BPE vocab. Use `config.json`'s value (128,256) for embedding table sizing.

#### Chat template structure

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>\n\n{system content}<|eot_id|>
<|start_header_id|>user<|end_header_id|>\n\n{user content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n   ← model generates from here
```

The system turn is injected automatically by the template (with a "Cutting Knowledge Date" preamble). The engine must pass `add_generation_prompt=True` to append the open assistant header before generation.

---

## Phase 0.4 — PACE Environment Bootstrap

*Not yet completed — parallel track on PACE Phoenix.*

---

## Phase 0.5 — nanobind Round-Trip

*Not yet completed — depends on 0.4.*

---

## Phase 1 — Correctness ✅

**Completed:** 2026-06-04

### What was built

| Path | Purpose |
|------|---------|
| `tests/oracle.py` | Runs HF Llama 3.2 1B in fp32, hooks into every layer, saves captured states to pickle files |
| `tests/fixtures/oracle_short.pkl` | Captured states for prompt "Hello, I am" (gitignored) |
| `tests/fixtures/oracle_medium.pkl` | Captured states for medium prompt (gitignored) |
| `tests/conftest.py` | Shared pytest fixtures — weights/config/oracle loaded once per session |
| `engine/loader.py` | Full implementation: safetensors → fp32 numpy dict, shape assertions, tied lm_head alias |
| `engine/components.py` | RMSNorm, RoPE (Llama3 scaling), GQA attention, SwiGLU FFN — all pure NumPy functions |
| `engine/model.py` | `LlamaModel.forward()`, `forward_debug()`, `greedy_decode()` |
| `tests/test_loader.py` | 4 tests — key presence, shape correctness, dtype, tied alias |
| `tests/test_components.py` | 11 tests — each component diffed against HF reference |
| `tests/test_forward.py` | 8 tests — embed exact match, layer-by-layer diff, final logit diff + argmax |
| `tests/test_decode.py` | 3 tests — 32 greedy tokens bit-identical to HF on two prompts, EOS stop |

**Test results: 26/26 passing.**

---

### Component deep-dives

#### engine/loader.py — Weight Loader

Loads 146 tensors from `model.safetensors`, casts bfloat16 → float32, asserts every shape against a config-derived table. If any tensor is missing or wrong shape, raises `ValueError` with the exact tensor name.

Key detail: `tie_word_embeddings=True` means no `lm_head.weight` exists in the file. The loader adds it as a Python alias (`weights["lm_head.weight"] = weights["model.embed_tokens.weight"]`) — same array object in memory, no copy. This means `x @ weights["lm_head.weight"].T` at the end of forward is just `x @ embed.T`.

---

#### engine/components.py — RMSNorm

```
output = x * weight / sqrt(mean(x²) + eps)
```

eps is **inside** the sqrt. Putting it outside is the classic bug — it changes which values get normalized how. Llama uses `rms_norm_eps=1e-5`. The formula normalizes by the root-mean-square of `x`, then scales by the learned `weight` vector (one per hidden dimension).

Why RMSNorm instead of LayerNorm? RMSNorm drops the mean-centering step (only normalizes by RMS, not mean+RMS). Faster and works just as well for transformers.

---

#### engine/components.py — RoPE (Rotary Position Embedding)

RoPE encodes position by *rotating* query and key vectors by a position-dependent angle. This means the dot product `q·k` naturally encodes the *relative* distance between positions — without needing separate learned position embeddings.

**How it works:**
1. Split each head's vector into pairs of dimensions
2. Each pair `(x₁, x₂)` at position `p` gets rotated by angle `p × θ_i` where `θ_i` depends on the dimension index
3. After rotation: `q·k` at positions `p` and `q` depends only on `p-q` (relative distance)

**Llama3 scaling (critical):** This model extends context from 8192 → 131072 tokens using frequency-dependent scaling:
- High-frequency components (short wavelengths): unscaled — preserve local patterns
- Low-frequency components (long wavelengths): divided by 32 — extend long-range reach  
- Between: smooth blend

Without this scaling, `rope_theta=500000` alone produces wrong encodings beyond position ~8192.

**Half-rotation layout:** Llama rotates `[-x[d/2:], x[:d/2]]` against `[x[:d/2], x[d/2:]]`. NOT interleaved. Getting this wrong produces incorrect attention and the model generates garbage.

---

#### engine/components.py — GQA Attention (Grouped Query Attention)

Standard multi-head attention uses one K/V head per Q head. GQA uses fewer K/V heads, with each K/V head shared across a group of Q heads. Llama 3.2 1B: 32 Q heads, 8 K/V heads → group ratio 4.

**Why GQA?** K/V cache is the memory bottleneck during generation. With 8 K/V heads instead of 32, KV cache is 4× smaller. Quality barely degrades because K/V heads share information well across similar Q heads.

**Forward pass:**
```
q = x @ q_proj.T  → reshape (seq, 32, 64)    # 32 query heads
k = x @ k_proj.T  → reshape (seq, 8,  64)    # 8 KV heads
v = x @ v_proj.T  → reshape (seq, 8,  64)

apply RoPE to q and k

k = repeat(k, 4, axis=1)  → (seq, 32, 64)   # broadcast each KV head to 4 Q heads
v = repeat(v, 4, axis=1)

scores = (q @ k.T) / sqrt(64)                # scaled dot product
scores += causal_mask                         # -inf for future positions
scores = softmax(scores)

out = scores @ v → reshape (seq, 2048)
out = out @ o_proj.T
```

**Causal mask:** upper triangle = −∞ so softmax gives 0 weight to future tokens. This is what makes autoregressive generation work — each token can only attend to itself and previous tokens.

---

#### engine/components.py — SwiGLU FFN

```
gate = silu(x @ gate_proj.T)   # gating signal
up   = x @ up_proj.T           # value signal
out  = (gate * up) @ down_proj.T
```

SiLU: `x * sigmoid(x)` = `x / (1 + e^(-x))`. A smooth activation function.

The "GLU" (Gated Linear Unit) part: `gate * up` means the gate learns *which* features of `up` to pass through. This is more expressive than a simple `silu(x @ W)`.

Why 8192 intermediate dim (4× hidden)? Standard transformer FFN uses 4× expansion to create a richer feature space for each token, then projects back down.

---

#### engine/model.py — LlamaModel

Wires all components into the full transformer forward pass. Pre-norm residual structure (norm BEFORE attention/FFN, not after):

```
x = embed[token_ids]                          # lookup
for each of 16 layers:
    h = rms_norm(x, input_layernorm)          # norm first
    h = gqa_attention(h, ...)                 # attention
    x = x + h                                 # residual 1 — add back to un-normed x
    
    h = rms_norm(x, post_attn_layernorm)      # norm first
    h = swiglu_ffn(h, ...)                    # FFN
    x = x + h                                 # residual 2

x = rms_norm(x, final_norm)
logits = x @ embed.T                          # tied lm_head
```

**Why residuals?** Without them, gradients vanish in deep networks (training problem). At inference, residuals let signal flow through unchanged layers — early layers build coarse features, later layers refine.

**Why pre-norm?** Post-norm (original transformer) applied norm after residual, causing training instability at large scale. Pre-norm stabilizes training and is standard in all modern LLMs.

`forward_debug()` is identical to `forward()` but also captures every intermediate state into a dict. Used by `test_forward.py` for layer-by-layer diffs against the oracle. Not used in production path.

---

#### engine/model.py — greedy_decode()

```python
for _ in range(max_tokens):
    logits = model.forward(ids)       # full forward pass on all tokens so far
    next_id = argmax(logits[-1])      # pick highest-probability next token
    ids.append(next_id)
    if next_id in {128001, 128008, 128009}:
        break
```

Phase 1 recomputes the entire forward pass from scratch each step — O(n²) work total for n tokens. This is correct but slow. Phase 2 fixes this with a KV cache: cache the K/V tensors from previous steps, only compute attention for the new token.

---

### Test infrastructure

#### tests/oracle.py

Loads HF Llama 3.2 1B in fp32, registers forward hooks, runs one forward pass, saves intermediate states to pickle. Runs once (~60s on CPU). All downstream tests load from pickle — no HF model re-loads needed.

**What hooks capture:**
- `post_embed`: output of embedding lookup (seq, 2048)
- `layer_N_post_attn`: input to `post_attention_layernorm` = hidden state after attention + residual 1
- `layer_N_post_ffn`: output of full decoder layer = hidden state after FFN + residual 2  
- `post_final_norm`: output of final RMSNorm
- `logits`: output of lm_head (seq, 128256)
- `greedy_ids`: 32 greedily sampled token IDs

**Why pickle and not re-run every test?** Loading HF model takes ~5s and running forward takes ~10s. With 26 tests, that's 4+ minutes of HF overhead per `pytest`. Fixtures make the full suite run in ~2 minutes.

#### tests/conftest.py

`scope="session"` fixtures: weights and oracle data load once per `pytest` run, not once per test. All 26 tests share the same loaded objects.

#### Tolerance ladder (validated)

| Check | atol | Actual max diff seen |
|-------|------|---------------------|
| embed lookup | 1e-6 | ~0 (exact) |
| rms_norm | 1e-5 | ~3e-7 |
| rope tables | 1e-5 | ~1e-6 |
| apply_rope | 2e-5 | ~1.2e-5 at pos 64 |
| gqa_attention | 1e-4 | ~3e-5 |
| swiglu_ffn | 1e-4 | ~2e-5 |
| per-layer hidden states | 5e-3 | ~2.2e-3 (fp32 amplification) |
| final logits | 1e-3 | ~5e-4 |
| argmax | exact | 0 mismatches |
| greedy 32 tokens | exact | 0 mismatches |

**Why do intermediate layer states have higher error (5e-3) than final logits (1e-3)?** fp32 matrix multiplications introduce ~1e-5 relative error per op. The FFN has a 2048→8192→2048 expansion, which amplifies a 7e-5 input error ~30× to ~2e-3. These errors partly cancel across 16 layers (they're not all in the same direction), so the final logit error is smaller than the worst intermediate error.

---

### Phase 1 milestone

✅ Coherent text from real Llama 3.2 1B weights  
✅ Per-layer hidden states match HF within tolerance  
✅ Final logits match HF within 1e-3, argmax exact at every position  
✅ 32 greedy generated tokens bit-identical to HF `model.generate(do_sample=False)`  

The make-or-break milestone is complete. Every component is proven correct against a HF oracle. All tests persist as regression suite through Phase 2+.

---

## Phase 2 — Usable Engine ✅

**Completed:** 2026-06-04

### What was built

| Path | Purpose |
|------|---------|
| `engine/cache.py` | `KVCache` class — pre-allocated K/V buffers for all layers |
| `engine/components.py` | `gqa_attention` extended with optional `kv_cache` + `layer_idx` params |
| `engine/model.py` | `LlamaModel.prefill()` and `LlamaModel.decode_step()` added alongside unchanged `forward()` |
| `engine/sampler.py` | `greedy()`, `sample()` (temp + top-k + top-p), `get_sampler()` factory |
| `engine/scheduler.py` | `generate()` — prefill-then-decode generator loop using KV cache |
| `engine/cli.py` | `llm-generate` CLI — argparse, chat template, streaming stdout |
| `engine/server.py` | FastAPI `/v1/chat/completions` — streaming (SSE) and non-streaming modes |
| `tests/test_sampler.py` | 13 unit tests — greedy, temperature, top-k, top-p, seed reproducibility |
| `tests/test_cache.py` | 9 unit tests — shape, dtype, pos tracking, write/read correctness |
| `tests/test_generate.py` | 5 tests — cache vs no-cache identity (slow), KV pos tracking (fast) |
| `tests/test_server.py` | 2 smoke tests — non-stream + stream HTTP responses (slow) |

**Test results: 45/45 passing** (fast suite — excludes slow-marked tests).

---

### How to run tests

```bash
# Fast suite (~8 min) — all Phase 1 + Phase 2 unit tests, no token generation
pytest tests/test_loader.py tests/test_components.py tests/test_forward.py \
       tests/test_sampler.py tests/test_cache.py -v

# Slow suite — KV cache identity + HTTP smoke test (requires full generation)
pytest -m slow -v

# Everything except the old no-cache decode tests (recommended daily)
pytest -m "not slow" --ignore=tests/test_decode.py -v
```

---

### Component deep-dives

#### engine/cache.py — KVCache

Pre-allocates two NumPy arrays of shape `(n_layers, max_seq, n_kv_heads, head_dim)` — one for K, one for V. `pos` tracks the next write position.

```python
cache.k[layer, cache.pos : cache.pos + n] = k_new   # write
k_full = cache.k[layer, :cache.pos + n]              # read full history
cache.advance(n)                                     # move write pointer
```

`gqa_attention` writes directly into the cache arrays during the forward pass. The caller (`prefill` or `decode_step`) calls `advance()` after all layers complete — not per layer.

**Why pre-allocate to max_seq?** Avoids reallocation during generation. Every decode step is a fixed-cost write at `cache.pos`, no copies. Memory cost: `2 × n_layers × max_seq × n_kv_heads × head_dim × 4 bytes` = for Llama 3.2 1B at max_seq=2048: `2 × 16 × 2048 × 8 × 64 × 4 = 268 MB`. Acceptable on CPU and trivial on A100.

---

#### engine/components.py — gqa_attention with KV cache

The no-cache path is **unchanged** — all Phase 1 tests still pass against it.

When `kv_cache` is provided:
1. Compute new K/V for current tokens (1 token for decode, full seq for prefill)
2. Write new K/V into cache at `cache.pos..cache.pos+seq`
3. Read full K/V history `cache.k[layer, :cache.pos+seq]` for attention
4. Causal mask: only applied when `seq > 1` (prefill). For decode (`seq=1`), all KV positions are already in the past — no mask needed.

```python
if seq > 1:
    offset = kv_seq - seq   # prior tokens already in cache (0 for fresh prefill)
    mask = np.triu(np.full((seq, kv_seq), -inf), k=offset + 1)
    scores += mask[np.newaxis]
```

`np.triu(..., k=offset+1)` makes query `qi` attend to KV positions `0..cache.pos+qi` — correct for both fresh prefill (offset=0) and any pre-filled cache.

---

#### engine/model.py — prefill() and decode_step()

`forward()` is unchanged. Two new methods add the cached path:

**`prefill(token_ids, kv_cache)`**
- Runs same layer loop as `forward()` but passes `kv_cache` and `layer_idx` to `gqa_attention`
- Returns logits for the **last token only** — shape `(vocab,)` — since that's the one we sample from
- Advances `cache.pos` by `len(token_ids)` after all layers complete

**`decode_step(token_id, kv_cache)`**
- `x` shape: `(1, hidden)` — single new token
- `positions`: `[cache.pos]` — the absolute position for RoPE
- Attends over full KV history `0..cache.pos` (read from cache) + writes new K/V
- Returns logits — shape `(vocab,)`
- Advances `cache.pos` by 1

**Why return just `(vocab,)` not `(seq, vocab)`?** Callers only need the next-token logits. Returning the full `(seq, vocab)` matrix would waste memory and the sampler would need to index `[-1]` anyway.

---

#### engine/sampler.py — Sampling

All pure functions, no state (except the seeded RNG in `get_sampler`).

**Pipeline order:** temperature → top-k → softmax → top-p → sample

```python
logits /= temperature                          # temperature scaling
# top-k: zero out all but k highest logits
topk_idx = np.argpartition(logits, -k)[-k:]
mask = np.full_like(logits, -inf)
mask[topk_idx] = logits[topk_idx]
logits = mask
# softmax
probs = softmax(logits)
# top-p: keep minimal set where cumsum >= p
sorted_idx = np.argsort(probs)[::-1]
cutoff = np.searchsorted(cumsum(probs[sorted_idx]), p) + 1
# sample
token = rng.choice(vocab, p=probs)
```

**Why temperature before softmax?** `logits / T` before softmax is equivalent to `(probs^(1/T)) / Z`. Low T sharpens the distribution (→ greedy), high T flattens it (→ uniform). Applying after softmax loses this equivalence.

**`get_sampler(temp, top_k, top_p, seed)`** returns a closure that bundles a seeded `np.random.Generator`. Same seed → bit-identical token sequences across runs.

---

#### engine/scheduler.py — generate()

Single-function generator replacing the no-cache `greedy_decode` from Phase 1:

```python
cache = KVCache(...)
logits = model.prefill(token_ids, cache)   # one forward pass, writes all prompt K/V
next_id = sampler_fn(logits)
yield next_id

for _ in range(max_tokens - 1):
    logits = model.decode_step(next_id, cache)  # one forward pass, single token
    next_id = sampler_fn(logits)
    yield next_id
    if next_id in EOS_IDS:
        break
```

**Why a generator?** Callers get tokens as they arrive — no waiting for full generation. The CLI flushes each token immediately; the server streams each token as an SSE event. Both use the same `generate()` — no duplication.

**Why prefill returns only last-token logits?** The prompt tokens are not sampled — only the token following the prompt is. Returning the full `(seq, vocab)` matrix wastes memory; we only need `logits[-1]`.

---

#### engine/cli.py — CLI

```bash
llm-generate --prompt "What is the capital of France?" --max-tokens 100 --temp 0.8 --top-k 50 --seed 42
```

Applies HF chat template before tokenizing, so the model sees the correct `<|start_header_id|>user<|end_header_id|>` wrapping. Streams decoded text to stdout token-by-token with `flush=True`.

---

#### engine/server.py — HTTP Server

FastAPI app. Model loads once at startup (`@app.on_event("startup")`), reused for every request.

**Streaming (`stream=true`):** Returns `StreamingResponse` with `media_type="text/event-stream"`. Each token yields:
```
data: {"choices": [{"delta": {"content": "token"}, "finish_reason": null}]}
```
Final chunk has `"finish_reason": "stop"` then `data: [DONE]`.

**Non-streaming (`stream=false`):** Collects all tokens, returns single JSON matching OpenAI schema.

```bash
# Start server
uvicorn engine.server:app --port 8000

# Non-streaming
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20, "stream": false}'

# Streaming
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20, "stream": true}'
```

---

### Phase 2 milestone

✅ KV cache pre-allocated and wired into attention — O(n) decode steps instead of O(n²)  
✅ `prefill()` + `decode_step()` on `LlamaModel` — Phase 1 `forward()` unchanged  
✅ All Phase 1 tests still passing — no regressions  
✅ Greedy, temperature, top-k, top-p sampling — all unit tested  
✅ `generate()` generator — prefill-then-decode with KV cache  
✅ CLI — streaming token output to stdout  
✅ HTTP server — OpenAI-compatible `/v1/chat/completions`, SSE streaming + non-streaming  

**Still to verify (run separately — slow):**
- `pytest -m slow -v` — cache vs no-cache identity (32 tokens bit-identical) + HTTP smoke test

---

## Phase 3 — Baseline Benchmarks

*Not yet started.*

---

## Phase 4 — Differentiators

*Not yet started.*

---

## Phase 5 — Polish

*Not yet started.*

---

## Reference: The From-Scratch Boundary

Per PRD §2, this is what counts as "from-scratch" vs "library is fine":

| Component | From-scratch (you implement) | Library (use freely) |
|-----------|-----------------------------|--------------------|
| Embedding lookup | ✅ | |
| RMSNorm | ✅ | |
| RoPE | ✅ | |
| GQA attention | ✅ | |
| SwiGLU FFN | ✅ | |
| KV cache | ✅ | |
| Scheduler / batching | ✅ | |
| Paged memory manager | ✅ | |
| Quantization logic | ✅ | |
| CUDA attention kernel | ✅ | |
| Array storage (NumPy/torch tensors) | | ✅ |
| GEMM (cuBLAS) | | ✅ |
| Tokenizer | | ✅ (HF tokenizers) |
| Weight parsing | | ✅ (safetensors) |

This boundary is what makes the project interview-credible: you are not reimplementing BLAS, you are implementing the model logic and the serving system.

---

## Reference: Llama 3.2 1B Architecture

Source: `weights/config.json`. Full details in `notes/model_config.md`.

- `hidden_size` — 2048
- `num_attention_heads` — 32
- `num_key_value_heads` — 8
- `num_hidden_layers` — 16
- `intermediate_size` — 8192
- `rope_theta` — 500000.0 (plus Llama3 scaling — see notes)
- `rms_norm_eps` — 1e-05
- `vocab_size` — 128256
- `tie_word_embeddings` — true (no separate lm_head tensor)
- GQA group ratio — 4 (32 query heads / 8 KV heads)
