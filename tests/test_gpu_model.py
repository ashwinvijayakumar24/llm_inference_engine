"""
End-to-end GPU model tests — require real weights and CUDA.
Marked @slow: run once on PACE to validate, not in daily loop.

    pytest tests/test_gpu_model.py -v -m slow

Note: loads GPU weights only (fp16, ~2.4GB VRAM).
CPU vs GPU argmax correctness is covered by test_components_gpu.py fast tests.
"""

import numpy as np
import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
slow = pytest.mark.slow

WEIGHTS_PATH = "weights"
TOKEN_IDS    = [128000, 9906, 358, 1097]   # <bos> Hello I am
VOCAB_SIZE   = 128256
EOS_IDS      = {128001, 128008, 128009}


def _load_gpu_model():
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU
    config  = load_config(WEIGHTS_PATH)
    weights = load_weights_gpu(WEIGHTS_PATH, config)
    return LlamaModelGPU(weights, config), config


@cuda_only
@slow
def test_gpu_prefill_returns_valid_logits():
    """Prefill produces finite logits of correct shape; argmax is a valid token ID."""
    model, config = _load_gpu_model()
    cache  = model.make_cache(2048)
    logits = model.prefill(TOKEN_IDS, cache)

    assert logits.shape == (VOCAB_SIZE,), f"Wrong logits shape: {logits.shape}"
    assert np.all(np.isfinite(logits)), "Logits contain NaN or Inf"
    next_id = int(np.argmax(logits))
    assert 0 <= next_id < VOCAB_SIZE, f"argmax {next_id} out of vocab range"
    print(f"\n  next token id: {next_id}")


@cuda_only
@slow
def test_gpu_generates_5_valid_tokens():
    """Generate 5 tokens greedily — all must be valid token IDs, no crash."""
    from engine.sampler import greedy
    from engine.scheduler import generate

    model, _ = _load_gpu_model()
    tokens   = list(generate(model, TOKEN_IDS, greedy, max_tokens=5))

    assert 1 <= len(tokens) <= 5, f"Expected 1-5 tokens, got {len(tokens)}"
    for tok in tokens:
        assert 0 <= tok < VOCAB_SIZE, f"Token {tok} out of vocab range"
    print(f"\n  generated tokens: {tokens}")


@cuda_only
@slow
def test_gpu_decode_step_advances_cache():
    """Cache position advances correctly after prefill + decode steps."""
    model, _ = _load_gpu_model()
    cache    = model.make_cache(2048)

    model.prefill(TOKEN_IDS, cache)
    assert cache.pos == len(TOKEN_IDS), f"Expected pos={len(TOKEN_IDS)}, got {cache.pos}"

    logits = model.decode_step(TOKEN_IDS[-1], cache)
    assert cache.pos == len(TOKEN_IDS) + 1
    assert logits.shape == (VOCAB_SIZE,)
    assert np.all(np.isfinite(logits))
