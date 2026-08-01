"""
From-scratch LLM inference engine for Llama 3.2 1B.

Public API — everything listed in ``__all__`` is stable for downstream
consumers. Anything else is an implementation detail and may change.

    from engine import LlamaModelGPU, generate, get_sampler, load_config, load_weights_gpu

    config  = load_config("weights")
    weights = load_weights_gpu("weights", config)
    model   = LlamaModelGPU(weights, config)

    for token_id in generate(model, prompt_ids, get_sampler(temp=0.0), max_tokens=64):
        ...

Imports are lazy (PEP 562) so that ``import engine`` does not pull in torch or
CUDA — the NumPy reference path stays usable on a CPU-only machine.
"""

__version__ = "0.1.0"

__all__ = [
    # Weight loading
    "load_config",
    "load_weights",
    "load_weights_gpu",
    "load_weights_gpu_quant",
    # Models — CPU fp32 reference and GPU fp16
    "LlamaModel",
    "LlamaModelGPU",
    # KV cache
    "KVCache",
    "KVCacheGPU",
    # Pluggable attention/KV backend — the seam a serving layer injects through.
    # Importing these does NOT pull in torch (they are a Protocol and a dataclass
    # whose tensor annotations are TYPE_CHECKING-only), so the CPU-only path is
    # unaffected.
    "AttentionBackend",
    "BatchMeta",
    # Generation
    "generate",
    "greedy",
    "get_sampler",
    # Quantization
    "QuantWeight",
]

# public name -> submodule that defines it
_EXPORTS = {
    "load_config":            "engine.loader",
    "load_weights":           "engine.loader",
    "load_weights_gpu":       "engine.loader",
    "load_weights_gpu_quant": "engine.loader",
    "LlamaModel":             "engine.model",
    "LlamaModelGPU":          "engine.model_gpu",
    "KVCache":                "engine.cache",
    "KVCacheGPU":             "engine.cache",
    "AttentionBackend":       "engine.attention_backend",
    "BatchMeta":              "engine.attention_backend",
    "generate":               "engine.scheduler",
    "greedy":                 "engine.sampler",
    "get_sampler":            "engine.sampler",
    "QuantWeight":            "engine.quant",
}


def __getattr__(name: str):
    """Resolve a public name to its submodule on first access (PEP 562)."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'engine' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)
