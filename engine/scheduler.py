"""Generation loop (single-request, Phase 2). Batched continuous scheduling in Phase 4."""

from typing import Callable, Generator

import numpy as np

from engine.cache import KVCache
from engine.model import EOS_IDS


def generate(
    model,
    token_ids: list[int],
    sampler_fn: Callable[[np.ndarray], int],
    max_tokens: int = 256,
    max_seq: int = 2048,
) -> Generator[int, None, None]:
    """
    Prefill the prompt then decode one token at a time using KV cache.

    Yields each generated token ID (including EOS if one is produced).
    Stops when EOS is yielded or max_tokens is reached.

    sampler_fn: callable that takes logits (vocab,) and returns a token ID int.
    """
    cache = KVCache(
        n_layers   = model.n_layers,
        max_seq    = max_seq,
        n_kv_heads = model.n_kv,
        head_dim   = model.head_dim,
    )

    # Prefill: process the full prompt, get logits for the last position
    logits  = model.prefill(token_ids, cache)
    next_id = sampler_fn(logits)
    yield next_id
    if next_id in EOS_IDS:
        return

    # Decode loop: one new token per step
    for _ in range(max_tokens - 1):
        logits  = model.decode_step(next_id, cache)
        next_id = sampler_fn(logits)
        yield next_id
        if next_id in EOS_IDS:
            break
