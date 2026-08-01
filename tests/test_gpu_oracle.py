"""
Model-level correctness gate for the GPU fp16 path.

WHY THIS FILE EXISTS
--------------------
Until now nothing verified that the GPU model produces *correct* output.

  * tests/test_forward.py and tests/test_decode.py compare against the
    HuggingFace oracle, but they exercise the CPU fp32 NumPy model.
  * tests/test_gpu_model.py runs the GPU model and asserts only that logits are
    finite, correctly shaped, and have an in-range argmax. A model returning
    confident nonsense passes every one of those checks.
  * tests/test_components_gpu.py compares GPU vs CPU components, but on synthetic
    random tensors with toy dimensions (seq=6, 4 heads, head_dim=8) scaled by
    0.02 — never on real weights, never end to end.

So every published correctness claim describes the CPU path, while every
published performance number describes the GPU path. That gap is fine for a
batch-1 benchmark harness. It is not fine as the foundation for a paged,
batched, preemptible serving layer: when batched output later disagrees with
single-sequence output, there must be a known-good reference to bisect against,
or the bug is unattributable across the allocator, the batching, the paged
attention, and a pre-existing GPU-port defect.

WHY TOKEN EQUALITY, NOT LOGIT DISTANCE
--------------------------------------
The CPU tests use tight logit tolerances (1e-3, test_forward.py:113) because
they compare fp32 against fp32. Here fp16 GPU is compared against fp32 CPU,
where existing component tests already need atol=1e-2 (test_components_gpu.py:83)
and accumulate error across 16 layers. A logit tolerance strict enough to be
meaningful would fail for legitimate reasons; one loose enough to pass would
catch nothing.

Greedy token identity is the right gate: it is what a user actually observes,
it is exactly the property batching and paging must preserve, and it either
holds or it does not.

    pytest tests/test_gpu_oracle.py -v            # needs CUDA + weights
"""

import numpy as np
import pytest
import torch

from tests.oracle import load_fixture

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
slow = pytest.mark.slow

WEIGHTS_PATH = "weights"
VOCAB_SIZE = 128256

# How many greedy tokens must match the fp32 oracle exactly.
#
# The oracle stores 32 (tests/oracle.py:176-183). We gate on a prefix rather
# than all 32 because fp16-vs-fp32 divergence is a legitimate possibility once
# two logits fall within fp16 resolution of each other — at which point the
# argmax can flip for numerical, not logical, reasons.
#
# This number is a MEASURED value, not an aspiration. The test reports the true
# divergence point on failure; if the real number is lower, record it here with
# the evidence rather than loosening the assertion into meaninglessness. A
# divergence at token 3 means something is broken. A divergence at token 28
# means fp16 is fp16.
GREEDY_PREFIX_MUST_MATCH = 16


@pytest.fixture(scope="module")
def gpu_model():
    """Load fp16 GPU weights once for the whole module (~2.4 GB VRAM)."""
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU

    config = load_config(WEIGHTS_PATH)
    weights = load_weights_gpu(WEIGHTS_PATH, config)
    return LlamaModelGPU(weights, config), config


def _greedy_gpu(model, token_ids, max_tokens):
    """Greedy decode on the GPU model via the engine's own generation loop."""
    from engine.sampler import greedy
    from engine.scheduler import generate

    return list(generate(model, list(token_ids), greedy, max_tokens=max_tokens))


def _first_divergence(a, b):
    """Index of the first differing element, or None if one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@cuda_only
@slow
@pytest.mark.parametrize("key", ["short", "medium"])
def test_gpu_greedy_matches_hf_oracle(gpu_model, key):
    """
    THE GATE. GPU fp16 greedy tokens must match HuggingFace fp32 for a prefix.

    Everything the serving layer later claims about correctness — batch
    invariance, preemption equality, cache-on/off equality — is a claim that
    output is unchanged relative to *this*.
    """
    model, _ = gpu_model
    fixture = load_fixture(key)
    prompt = list(fixture["token_ids"])
    expected = list(fixture["greedy_ids"])

    got = _greedy_gpu(model, prompt, max_tokens=len(expected))

    n = min(GREEDY_PREFIX_MUST_MATCH, len(expected), len(got))
    div = _first_divergence(got[:n], expected[:n])

    if div is not None:
        full_div = _first_divergence(got, expected)
        pytest.fail(
            f"GPU greedy output diverges from the fp32 HF oracle at token {div} "
            f"(required prefix: {n}).\n"
            f"  expected[{div}] = {expected[div]}\n"
            f"  got[{div}]      = {got[div]}\n"
            f"  first divergence over the full {len(expected)}-token sequence: {full_div}\n"
            f"  expected: {expected[:n]}\n"
            f"  got:      {got[:n]}\n"
            "An early divergence (< ~8 tokens) indicates a real defect in the GPU "
            "port, not fp16 rounding."
        )

    print(f"\n  [{key}] {n} greedy tokens match the fp32 oracle exactly")


@cuda_only
@slow
@pytest.mark.parametrize("key", ["short", "medium"])
def test_gpu_matches_cpu_reference(gpu_model, key):
    """
    GPU fp16 vs CPU fp32 — the SAME engine, both paths.

    Distinct from the oracle test above, and worth having separately: this one
    isolates "the GPU port is wrong" from "the engine is wrong". If the oracle
    test fails and this one passes, both engine paths agree and disagree with
    HuggingFace — a modelling bug. If this one fails, the GPU port itself
    diverges from a path already validated to <1e-3 against HF.
    """
    from engine.loader import load_config, load_weights
    from engine.model import LlamaModel

    model_gpu, _ = gpu_model
    fixture = load_fixture(key)
    prompt = list(fixture["token_ids"])
    n_tokens = GREEDY_PREFIX_MUST_MATCH

    config = load_config(WEIGHTS_PATH)
    # NOTE: loading CPU fp32 and GPU fp16 weights in one process is heavy.
    # The GPU weights are already resident; the CPU model is fp32 host memory,
    # so this is RAM pressure, not VRAM pressure.
    model_cpu = LlamaModel(load_weights(WEIGHTS_PATH, config), config)

    got_gpu = _greedy_gpu(model_gpu, prompt, max_tokens=n_tokens)
    got_cpu = _greedy_gpu(model_cpu, prompt, max_tokens=n_tokens)

    div = _first_divergence(got_gpu, got_cpu)
    assert div is None, (
        f"GPU fp16 diverges from CPU fp32 (same engine) at token {div}:\n"
        f"  cpu: {got_cpu}\n"
        f"  gpu: {got_gpu}\n"
        "The CPU path is validated against HF to <1e-3, so this points at the "
        "GPU port specifically."
    )


@cuda_only
@slow
def test_gpu_prefill_logits_track_oracle(gpu_model):
    """
    Prefill logits at the final position should be *close* to the oracle's, and
    their argmax must match exactly.

    Softer than the greedy gate and useful for a different reason: if greedy
    output diverges, this distinguishes "logits are slightly off and one argmax
    flipped" from "logits are structurally wrong". The first is fp16; the second
    is a bug.
    """
    model, _ = gpu_model
    fixture = load_fixture("short")
    prompt = list(fixture["token_ids"])
    oracle_logits = np.asarray(fixture["logits"])[-1]      # final position, (vocab,)

    cache = model.make_cache(2048)
    got = model.prefill(prompt, cache)

    assert got.shape == (VOCAB_SIZE,)
    assert np.all(np.isfinite(got)), "GPU prefill produced NaN or Inf"
    assert int(np.argmax(got)) == int(np.argmax(oracle_logits)), (
        f"argmax mismatch: gpu={int(np.argmax(got))} oracle={int(np.argmax(oracle_logits))}"
    )

    # Top-10 agreement as a structural check: fp16 may reorder near-ties, but a
    # correct model cannot disagree about which tokens are plausible at all.
    top_gpu = set(np.argsort(got)[-10:].tolist())
    top_oracle = set(np.argsort(oracle_logits)[-10:].tolist())
    overlap = len(top_gpu & top_oracle)
    assert overlap >= 8, (
        f"Only {overlap}/10 of the top-10 tokens agree with the oracle. "
        "fp16 rounding reorders near-ties; it does not change which tokens are "
        "plausible."
    )

    max_abs = float(np.max(np.abs(got - oracle_logits)))
    print(f"\n  max |logit diff| vs fp32 oracle: {max_abs:.4f}, top-10 overlap {overlap}/10")


@cuda_only
@slow
def test_gpu_chunked_prefill_positions(gpu_model):
    """
    Prefilling a prompt in two chunks must equal prefilling it in one.

    Guards the position fix in model_gpu.prefill: RoPE positions now continue
    from kv_cache.pos instead of restarting at 0. Before the fix the second
    chunk was told it began at position 0, corrupting RoPE for every token in
    it — invisible in every existing test, because generate() calls prefill
    exactly once (engine/scheduler.py:26-33).

    The serving layer's chunked prefill depends entirely on this holding.
    """
    model, _ = gpu_model
    prompt = list(load_fixture("medium")["token_ids"])
    assert len(prompt) >= 4, "need a prompt long enough to split"
    split = len(prompt) // 2

    cache_one = model.make_cache(2048)
    logits_one = model.prefill(prompt, cache_one)

    cache_two = model.make_cache(2048)
    model.prefill(prompt[:split], cache_two)      # chunk 1
    logits_two = model.prefill(prompt[split:], cache_two)  # chunk 2, positions continue

    assert cache_one.pos == cache_two.pos == len(prompt)
    assert int(np.argmax(logits_one)) == int(np.argmax(logits_two)), (
        "Chunked prefill produced a different next token than single-shot prefill. "
        "RoPE positions are not continuing correctly across chunks."
    )

    max_abs = float(np.max(np.abs(logits_one - logits_two)))
    print(f"\n  chunked vs single-shot prefill, max |logit diff|: {max_abs:.5f}")
