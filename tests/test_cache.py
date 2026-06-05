"""
KV cache unit tests — shape, dtype, allocation, position tracking.
No model needed for most tests.
"""

import numpy as np
import pytest

from engine.cache import KVCache


N_LAYERS   = 4
MAX_SEQ    = 64
N_KV_HEADS = 8
HEAD_DIM   = 64


@pytest.fixture
def cache():
    return KVCache(N_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM)


def test_initial_shape(cache):
    assert cache.k.shape == (N_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM)
    assert cache.v.shape == (N_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM)


def test_initial_dtype(cache):
    assert cache.k.dtype == np.float32
    assert cache.v.dtype == np.float32


def test_initial_pos(cache):
    assert cache.pos == 0


def test_advance_single(cache):
    cache.advance(1)
    assert cache.pos == 1


def test_advance_multiple(cache):
    cache.advance(5)
    cache.advance(3)
    assert cache.pos == 8


def test_advance_default_is_one(cache):
    cache.advance()
    assert cache.pos == 1


def test_write_via_direct_slice(cache):
    # gqa_attention writes directly: cache.k[layer, pos:pos+n] = k_new
    k_new = np.ones((1, N_KV_HEADS, HEAD_DIM), dtype=np.float32) * 7.0
    v_new = np.ones((1, N_KV_HEADS, HEAD_DIM), dtype=np.float32) * 3.0
    layer = 0

    cache.k[layer, cache.pos:cache.pos + 1] = k_new
    cache.v[layer, cache.pos:cache.pos + 1] = v_new
    cache.advance(1)

    np.testing.assert_array_equal(cache.k[layer, 0], k_new[0])
    np.testing.assert_array_equal(cache.v[layer, 0], v_new[0])


def test_multiple_tokens_written(cache):
    seq = 5
    k_new = np.random.rand(seq, N_KV_HEADS, HEAD_DIM).astype(np.float32)
    v_new = np.random.rand(seq, N_KV_HEADS, HEAD_DIM).astype(np.float32)
    layer = 2

    cache.k[layer, :seq] = k_new
    cache.v[layer, :seq] = v_new
    cache.advance(seq)

    np.testing.assert_array_equal(cache.k[layer, :seq], k_new)
    assert cache.pos == seq


def test_unwritten_positions_are_zero(cache):
    # Write only position 0, rest should still be zero
    cache.k[0, 0] = 1.0
    cache.advance(1)
    np.testing.assert_array_equal(cache.k[0, 1:], 0.0)
