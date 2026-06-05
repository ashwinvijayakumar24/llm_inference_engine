"""Weight loading: safetensors → named fp32 numpy arrays with shape assertions."""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def load_config(weights_path: str) -> dict:
    with open(Path(weights_path) / "config.json") as f:
        return json.load(f)


def _expected_shapes(config: dict) -> dict[str, tuple]:
    H   = config["hidden_size"]
    NH  = config["num_attention_heads"]
    NKV = config["num_key_value_heads"]
    HD  = config["head_dim"]
    NL  = config["num_hidden_layers"]
    FF  = config["intermediate_size"]
    V   = config["vocab_size"]

    shapes: dict[str, tuple] = {
        "model.embed_tokens.weight": (V, H),
        "model.norm.weight": (H,),
    }
    for i in range(NL):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (NH * HD, H)
        shapes[f"{p}.self_attn.k_proj.weight"] = (NKV * HD, H)
        shapes[f"{p}.self_attn.v_proj.weight"] = (NKV * HD, H)
        shapes[f"{p}.self_attn.o_proj.weight"] = (H, NH * HD)
        shapes[f"{p}.mlp.gate_proj.weight"]    = (FF, H)
        shapes[f"{p}.mlp.up_proj.weight"]      = (FF, H)
        shapes[f"{p}.mlp.down_proj.weight"]    = (H, FF)
        shapes[f"{p}.input_layernorm.weight"]            = (H,)
        shapes[f"{p}.post_attention_layernorm.weight"]   = (H,)
    return shapes


def load_weights(weights_path: str, config: dict) -> dict[str, np.ndarray]:
    """
    Load all model weights from safetensors, cast to fp32.
    Asserts every tensor name and shape against config-derived table.
    Adds 'lm_head.weight' alias when tie_word_embeddings is true.
    """
    expected = _expected_shapes(config)
    safetensors_file = Path(weights_path) / "model.safetensors"

    weights: dict[str, np.ndarray] = {}

    with safe_open(str(safetensors_file), framework="pt") as f:
        loaded_keys = set(f.keys())

        for name, exp_shape in expected.items():
            if name not in loaded_keys:
                raise ValueError(f"Missing tensor in safetensors: {name}")
            t = f.get_tensor(name)
            actual_shape = tuple(t.shape)
            if actual_shape != exp_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': "
                    f"got {actual_shape}, expected {exp_shape}"
                )
            # Cast bfloat16 → float32; NumPy has no bfloat16
            weights[name] = t.float().numpy()

    # Tied lm_head: reuse embed weights (no copy — same array)
    if config.get("tie_word_embeddings", False):
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    elif "lm_head.weight" in loaded_keys:
        with safe_open(str(safetensors_file), framework="pt") as f:
            t = f.get_tensor("lm_head.weight")
            weights["lm_head.weight"] = t.float().numpy()
    else:
        raise ValueError("lm_head.weight missing and tie_word_embeddings is false")

    return weights


def load_weights_gpu(weights_path: str, config: dict, device: str = "cuda:0") -> dict:
    """
    Load model weights as fp16 torch tensors on the given device.
    Skips shape assertions (load_weights already validates on CPU).
    Handles tied lm_head the same way as load_weights.
    """
    import torch
    from safetensors import safe_open

    safetensors_file = Path(weights_path) / "model.safetensors"
    weights: dict = {}

    with safe_open(str(safetensors_file), framework="pt") as f:
        loaded_keys = set(f.keys())
        for name in f.keys():
            weights[name] = f.get_tensor(name).half().to(device)

    if config.get("tie_word_embeddings", False):
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    elif "lm_head.weight" in loaded_keys:
        pass  # already loaded above
    else:
        raise ValueError("lm_head.weight missing and tie_word_embeddings is false")

    return weights
