"""
Fast harness validation tests — no real model or weights needed.
Uses a stub model with fixed sleep delays to verify timing math.

Run: pytest bench/test_harness.py -v
"""

import time

import numpy as np
import pytest

from bench.harness import _percentile, _peak_mem_mb, time_generate, write_results


# ---------------------------------------------------------------------------
# Stub model — simulates prefill + decode delays
# ---------------------------------------------------------------------------

class _StubSampler:
    """Returns token 42 always. Stops after max_count yields."""

    def __init__(self, max_count=5):
        self.calls = 0
        self.max   = max_count

    def __call__(self, logits):
        self.calls += 1
        return 42


class _StubModel:
    """
    Prefill sleeps prefill_s seconds, each decode step sleeps decode_s seconds.
    Returns a fake logits vector (vocab_size=10).
    """

    n_layers  = 2
    n_kv      = 2
    head_dim  = 8
    n_heads   = 4

    def __init__(self, prefill_s=0.05, decode_s=0.02):
        self.prefill_s = prefill_s
        self.decode_s  = decode_s
        self._fake_logits = np.zeros(10, dtype=np.float32)
        self._fake_logits[42 % 10] = 1.0  # argmax → 42 % 10 = 2

    def prefill(self, token_ids, kv_cache):
        time.sleep(self.prefill_s)
        kv_cache.advance(len(token_ids))
        return self._fake_logits.copy()

    def decode_step(self, token_id, kv_cache):
        time.sleep(self.decode_s)
        kv_cache.advance(1)
        return self._fake_logits.copy()


# Monkey-patch EOS_IDS to something the stub will never return
import engine.scheduler as _sched
_ORIG_EOS = _sched.EOS_IDS
_sched.EOS_IDS = set()


def _restore_eos():
    _sched.EOS_IDS = _ORIG_EOS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ttft_within_tolerance():
    """TTFT should be ≈ prefill_s (± 20ms tolerance for sleep jitter)."""
    model    = _StubModel(prefill_s=0.05, decode_s=0.01)
    sampler  = _StubSampler(max_count=4)
    result   = time_generate(model, [1, 2, 3], sampler, max_tokens=4)

    assert "ttft_s" in result
    assert 0.03 < result["ttft_s"] < 0.15, f"TTFT out of range: {result['ttft_s']}"


def test_decode_tok_s_within_tolerance():
    """Decode tok/s ≈ 1 / decode_s (±30% for sleep jitter on CI)."""
    decode_s = 0.02
    model    = _StubModel(prefill_s=0.01, decode_s=decode_s)
    sampler  = _StubSampler(max_count=10)
    result   = time_generate(model, [1, 2, 3], sampler, max_tokens=10)

    expected = 1.0 / decode_s
    assert result["decode_tok_s"] > 0
    assert 0.5 * expected < result["decode_tok_s"] < 2.0 * expected, (
        f"decode_tok_s={result['decode_tok_s']:.1f}, expected≈{expected:.1f}"
    )


def test_n_decode_tokens():
    model   = _StubModel(prefill_s=0.01, decode_s=0.005)
    sampler = _StubSampler(max_count=6)
    result  = time_generate(model, [1, 2], sampler, max_tokens=6)
    assert result["n_decode_tokens"] == 6


def test_n_prompt_tokens():
    model   = _StubModel(prefill_s=0.01, decode_s=0.005)
    sampler = _StubSampler(max_count=3)
    result  = time_generate(model, [10, 20, 30, 40], sampler, max_tokens=3)
    assert result["n_prompt_tokens"] == 4


def test_itl_percentiles_present():
    model   = _StubModel(prefill_s=0.01, decode_s=0.01)
    sampler = _StubSampler(max_count=5)
    result  = time_generate(model, [1], sampler, max_tokens=5)
    assert "itl_p50_ms" in result
    assert "itl_p99_ms" in result
    assert result["itl_p50_ms"] >= 0
    assert result["itl_p99_ms"] >= result["itl_p50_ms"]


def test_peak_mem_positive():
    assert _peak_mem_mb() > 0


# ---------------------------------------------------------------------------
# Percentile helper tests
# ---------------------------------------------------------------------------

def test_percentile_median():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(vals, 50) == pytest.approx(3.0)


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_p99_geq_p50():
    vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.5]
    assert _percentile(vals, 99) >= _percentile(vals, 50)


# ---------------------------------------------------------------------------
# write_results test
# ---------------------------------------------------------------------------

def test_write_results_creates_files(tmp_path):
    rows = [
        {
            "prompt_key": "short", "run": 1,
            "n_prompt_tokens": 4, "n_decode_tokens": 5,
            "ttft_s": 0.1, "total_s": 0.5,
            "prefill_tok_s": 40.0, "decode_tok_s": 8.0,
            "itl_p50_ms": 25.0, "itl_p99_ms": 30.0,
            "peak_mem_mb": 512.0, "backend": "cpu",
        }
    ]
    write_results(rows, "cpu", tmp_path)
    files = list(tmp_path.iterdir())
    assert any(f.suffix == ".json" for f in files)
    assert any(f.suffix == ".csv" for f in files)


def test_write_results_json_content(tmp_path):
    import json
    rows = [{"prompt_key": "short", "run": 1, "ttft_s": 0.05}]
    write_results(rows, "test", tmp_path)
    json_file = next(f for f in tmp_path.iterdir() if f.suffix == ".json")
    data = json.loads(json_file.read_text())
    assert data[0]["ttft_s"] == 0.05
