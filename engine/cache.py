"""KV cache: naive contiguous (Phase 2) → paged block allocator (Phase 4)."""


class NaiveKVCache:
    def __init__(self, n_layers: int, max_seq: int, n_kv_heads: int, head_dim: int):
        raise NotImplementedError
