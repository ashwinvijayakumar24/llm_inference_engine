"""
End-to-end generation test: generate() with KV cache vs greedy_decode() without cache.
Tokens must be bit-identical — this is the Phase 2 correctness gate.

Runtime: ~30–60s (full forward passes). Uses session-scoped model fixture.
"""

import pytest

from engine.model import greedy_decode
from engine.sampler import greedy
from engine.scheduler import generate
from tests.oracle import load_fixture


@pytest.fixture(scope="module")
def oracle_short():
    return load_fixture("short")


@pytest.fixture(scope="module")
def oracle_medium():
    return load_fixture("medium")


def _cached_greedy(model, token_ids, max_tokens):
    """Run generate() with greedy sampler, return list of token IDs."""
    return list(generate(model, token_ids, greedy, max_tokens=max_tokens))


def _no_cache_greedy(model, token_ids, max_tokens):
    """Run greedy_decode() (Phase 1 path), return list of token IDs."""
    return greedy_decode(model, token_ids, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Cache vs no-cache identity
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cache_matches_no_cache_short(model, oracle_short):
    token_ids = list(oracle_short["input_ids"])
    cached    = _cached_greedy(model, token_ids, max_tokens=32)
    no_cache  = _no_cache_greedy(model, token_ids, max_tokens=32)
    assert cached == no_cache, (
        f"Cache/no-cache mismatch at index {next(i for i,(a,b) in enumerate(zip(cached,no_cache)) if a!=b)}"
    )


@pytest.mark.slow
def test_cache_matches_no_cache_medium(model, oracle_medium):
    token_ids = list(oracle_medium["input_ids"])
    cached    = _cached_greedy(model, token_ids, max_tokens=32)
    no_cache  = _no_cache_greedy(model, token_ids, max_tokens=32)
    assert cached == no_cache, (
        f"Cache/no-cache mismatch at index {next(i for i,(a,b) in enumerate(zip(cached,no_cache)) if a!=b)}"
    )


@pytest.mark.slow
def test_cache_matches_hf_greedy_short(model, oracle_short):
    """Generated tokens must match HF reference (via oracle fixture)."""
    token_ids = list(oracle_short["input_ids"])
    cached    = _cached_greedy(model, token_ids, max_tokens=32)
    hf_ids    = list(oracle_short["greedy_ids"])
    assert cached == hf_ids, (
        f"Mismatch vs HF at index {next(i for i,(a,b) in enumerate(zip(cached,hf_ids)) if a!=b)}"
    )


# ---------------------------------------------------------------------------
# KV cache position tracking
# ---------------------------------------------------------------------------

def test_cache_pos_after_prefill(model, oracle_short):
    from engine.cache import KVCache

    token_ids = list(oracle_short["input_ids"])
    cache     = KVCache(model.n_layers, 2048, model.n_kv, model.head_dim)
    assert cache.pos == 0

    model.prefill(token_ids, cache)
    assert cache.pos == len(token_ids)


def test_cache_pos_after_decode(model, oracle_short):
    from engine.cache import KVCache

    token_ids = list(oracle_short["input_ids"])
    cache     = KVCache(model.n_layers, 2048, model.n_kv, model.head_dim)
    model.prefill(token_ids, cache)
    start_pos = cache.pos

    model.decode_step(token_ids[-1], cache)
    assert cache.pos == start_pos + 1
