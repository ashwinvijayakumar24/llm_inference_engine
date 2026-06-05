"""
GPU (torch) variant of LlamaModel — implements on PACE A100 (Phase 3.2).

Same prefill() / decode_step() interface as engine/model.py but:
- Weights loaded as fp16 torch tensors on cuda:0
- All NumPy ops replaced with torch ops
- KVCache uses torch tensors (fp16, GPU)

Does NOT touch engine/model.py — Phase 1 NumPy tests must stay passing.
"""


class LlamaModelGPU:
    def __init__(self, weights: dict, config: dict):
        raise NotImplementedError(
            "LlamaModelGPU not yet implemented. "
            "Implement on PACE after Phase 3.1 harness is validated on CPU."
        )

    def prefill(self, token_ids: list[int], kv_cache) -> "torch.Tensor":
        raise NotImplementedError

    def decode_step(self, token_id: int, kv_cache) -> "torch.Tensor":
        raise NotImplementedError
