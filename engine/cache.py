"""KV cache: naive contiguous (Phase 2) → paged block allocator (Phase 4)."""

import numpy as np


class KVCache:
    """
    Per-layer K/V buffers pre-allocated to max_seq length.

    gqa_attention writes directly into k[layer] and v[layer] using self.pos
    as the write offset. Call advance(n) after each forward pass to move pos.
    """

    def __init__(self, n_layers: int, max_seq: int, n_kv_heads: int, head_dim: int):
        self.k   = np.zeros((n_layers, max_seq, n_kv_heads, head_dim), dtype=np.float32)
        self.v   = np.zeros((n_layers, max_seq, n_kv_heads, head_dim), dtype=np.float32)
        self.pos = 0  # next write position

    def advance(self, n: int = 1) -> None:
        self.pos += n
