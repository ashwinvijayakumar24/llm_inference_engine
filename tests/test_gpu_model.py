"""
End-to-end GPU model tests — require real weights and CUDA.
Marked @slow: run once on PACE to validate, not in daily loop.

    pytest tests/test_gpu_model.py -v -m slow
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


@cuda_only
@slow
def test_gpu_prefill_argmax_matches_cpu():
    """GPU and CPU prefill must agree on the next token (exact argmax match)."""
    from engine.loader import load_config, load_weights, load_weights_gpu
    from engine.model import LlamaModel
    from engine.model_gpu import LlamaModelGPU
    from engine.cache import KVCache, KVCacheGPU

    config = load_config(WEIGHTS_PATH)

    w_cpu    = load_weights(WEIGHTS_PATH, config)
    model_c  = LlamaModel(w_cpu, config)
    cache_c  = KVCache(model_c.n_layers, 2048, model_c.n_kv, model_c.head_dim)
    logits_c = model_c.prefill(TOKEN_IDS, cache_c)

    w_gpu    = load_weights_gpu(WEIGHTS_PATH, config)
    model_g  = LlamaModelGPU(w_gpu, config)
    cache_g  = model_g.make_cache(2048)
    logits_g = model_g.prefill(TOKEN_IDS, cache_g)

    assert int(np.argmax(logits_c)) == int(np.argmax(logits_g)), (
        f"CPU argmax={int(np.argmax(logits_c))} != GPU argmax={int(np.argmax(logits_g))}"
    )


@cuda_only
@slow
def test_gpu_5_tokens_match_cpu():
    """5 greedily generated tokens must be identical on CPU and GPU."""
    from engine.loader import load_config, load_weights, load_weights_gpu
    from engine.model import LlamaModel
    from engine.model_gpu import LlamaModelGPU
    from engine.cache import KVCache, KVCacheGPU
    from engine.sampler import greedy
    from engine.scheduler import generate

    config = load_config(WEIGHTS_PATH)

    w_cpu   = load_weights(WEIGHTS_PATH, config)
    model_c = LlamaModel(w_cpu, config)
    cpu_tokens = list(generate(model_c, TOKEN_IDS, greedy, max_tokens=5))

    w_gpu   = load_weights_gpu(WEIGHTS_PATH, config)
    model_g = LlamaModelGPU(w_gpu, config)
    gpu_tokens = list(generate(model_g, TOKEN_IDS, greedy, max_tokens=5))

    assert cpu_tokens == gpu_tokens, (
        f"Token mismatch — CPU: {cpu_tokens}, GPU: {gpu_tokens}"
    )
