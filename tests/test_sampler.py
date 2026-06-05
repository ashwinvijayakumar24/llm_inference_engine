"""Unit tests for engine/sampler.py — fast, no model required."""

import numpy as np
import pytest

from engine.sampler import get_sampler, greedy, sample


# ---------------------------------------------------------------------------
# greedy
# ---------------------------------------------------------------------------

def test_greedy_argmax():
    logits = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    assert greedy(logits) == 3


def test_greedy_single():
    logits = np.array([1.0])
    assert greedy(logits) == 0


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def test_temperature_one_is_identity_distribution():
    # With T=1 and no top-k/top-p, sample should draw from unmodified softmax.
    # Run many draws and verify all valid indices returned.
    rng    = np.random.default_rng(0)
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    draws  = {sample(logits, temperature=1.0, rng=rng) for _ in range(200)}
    # All 4 indices should appear with enough draws
    assert draws == {0, 1, 2, 3}


def test_temperature_low_approaches_greedy():
    # T→0 (use a very small value) should almost always pick the argmax
    rng    = np.random.default_rng(42)
    logits = np.array([10.0, 1.0, 1.0, 1.0])
    draws  = [sample(logits, temperature=0.01, rng=rng) for _ in range(50)]
    # Should almost exclusively pick index 0
    assert draws.count(0) >= 48


def test_temperature_high_more_uniform():
    # T=100 should spread probability more uniformly
    rng    = np.random.default_rng(7)
    logits = np.array([5.0, 1.0, 1.0, 1.0])
    draws  = {sample(logits, temperature=100.0, rng=rng) for _ in range(500)}
    # At high temp all 4 tokens should appear
    assert len(draws) == 4


# ---------------------------------------------------------------------------
# top-k
# ---------------------------------------------------------------------------

def test_top_k_only_k_tokens_sampled():
    rng    = np.random.default_rng(0)
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    draws  = {sample(logits, temperature=1.0, top_k=2, rng=rng) for _ in range(300)}
    # Only the top-2 tokens (indices 3, 4) should appear
    assert draws == {3, 4}


def test_top_k_larger_than_vocab():
    # top_k > vocab_size should not crash, treats as top all
    rng    = np.random.default_rng(0)
    logits = np.array([1.0, 2.0, 3.0])
    draws  = {sample(logits, temperature=1.0, top_k=100, rng=rng) for _ in range(200)}
    assert draws == {0, 1, 2}


# ---------------------------------------------------------------------------
# top-p
# ---------------------------------------------------------------------------

def test_top_p_cumulative_prob():
    # With a peaked distribution and p=0.9, only the top token(s) should appear
    rng    = np.random.default_rng(0)
    # Token 0 has ~95% mass, rest share ~5%
    logits = np.array([10.0, 0.1, 0.1, 0.1])
    draws  = {sample(logits, temperature=1.0, top_p=0.9, rng=rng) for _ in range(200)}
    # Token 0 must appear (has >90% prob)
    assert 0 in draws


def test_top_p_one_no_filter():
    # top_p=1.0 means keep everything
    rng    = np.random.default_rng(0)
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    draws  = {sample(logits, temperature=1.0, top_p=1.0, rng=rng) for _ in range(400)}
    assert draws == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducible():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng_a  = np.random.default_rng(123)
    rng_b  = np.random.default_rng(123)
    draws_a = [sample(logits, temperature=1.0, rng=rng_a) for _ in range(20)]
    draws_b = [sample(logits, temperature=1.0, rng=rng_b) for _ in range(20)]
    assert draws_a == draws_b


def test_different_seeds_differ():
    logits  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    rng_a   = np.random.default_rng(1)
    rng_b   = np.random.default_rng(2)
    draws_a = [sample(logits, temperature=1.0, rng=rng_a) for _ in range(30)]
    draws_b = [sample(logits, temperature=1.0, rng=rng_b) for _ in range(30)]
    assert draws_a != draws_b


# ---------------------------------------------------------------------------
# get_sampler factory
# ---------------------------------------------------------------------------

def test_get_sampler_temp_zero_is_greedy():
    sampler = get_sampler(temp=0.0)
    logits  = np.array([0.1, 5.0, 0.2])
    assert sampler(logits) == 1   # argmax


def test_get_sampler_seeded_reproducible():
    logits   = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sampler1 = get_sampler(temp=1.0, seed=42)
    sampler2 = get_sampler(temp=1.0, seed=42)
    draws1   = [sampler1(logits) for _ in range(20)]
    draws2   = [sampler2(logits) for _ in range(20)]
    assert draws1 == draws2
