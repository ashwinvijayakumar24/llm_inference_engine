"""
Dumps all tensor names, shapes, and dtypes from the model safetensors file.
Output committed to notes/tensor_dump.txt.

Usage:
    python3 scripts/inspect_weights.py
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
SAFETENSORS_PATH = WEIGHTS_DIR / "model.safetensors"
CONFIG_PATH = WEIGHTS_DIR / "config.json"
OUTPUT_PATH = Path(__file__).parent.parent / "notes" / "tensor_dump.txt"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def dump_tensors():
    lines = []
    # Use torch backend — weights are bfloat16, unsupported by numpy safetensors backend
    with safe_open(str(SAFETENSORS_PATH), framework="pt") as f:
        keys = sorted(f.keys())
        for key in keys:
            t = f.get_tensor(key)
            lines.append(f"{key:80s}  shape={str(tuple(t.shape)):30s}  dtype={t.dtype}")
    return lines


def assert_shapes(config: dict, lines: list[str]):
    H = config["hidden_size"]
    NH = config["num_attention_heads"]
    NKV = config["num_key_value_heads"]
    HD = config["head_dim"]
    NL = config["num_hidden_layers"]
    FF = config["intermediate_size"]
    V = config["vocab_size"]
    tied = config["tie_word_embeddings"]

    shape_map: dict[str, tuple] = {}
    shape_map["model.embed_tokens.weight"] = (V, H)
    shape_map["model.norm.weight"] = (H,)
    for i in range(NL):
        p = f"model.layers.{i}"
        shape_map[f"{p}.self_attn.q_proj.weight"] = (NH * HD, H)
        shape_map[f"{p}.self_attn.k_proj.weight"] = (NKV * HD, H)
        shape_map[f"{p}.self_attn.v_proj.weight"] = (NKV * HD, H)
        shape_map[f"{p}.self_attn.o_proj.weight"] = (H, NH * HD)
        shape_map[f"{p}.mlp.gate_proj.weight"] = (FF, H)
        shape_map[f"{p}.mlp.up_proj.weight"] = (FF, H)
        shape_map[f"{p}.mlp.down_proj.weight"] = (H, FF)
        shape_map[f"{p}.input_layernorm.weight"] = (H,)
        shape_map[f"{p}.post_attention_layernorm.weight"] = (H,)

    if not tied:
        shape_map["lm_head.weight"] = (V, H)

    errors = []
    with safe_open(str(SAFETENSORS_PATH), framework="pt") as f:
        loaded_keys = set(f.keys())
        for name, expected_shape in shape_map.items():
            if name not in loaded_keys:
                errors.append(f"MISSING: {name}")
                continue
            t = f.get_tensor(name)
            if tuple(t.shape) != expected_shape:
                errors.append(
                    f"SHAPE MISMATCH: {name}  got={tuple(t.shape)}  expected={expected_shape}"
                )
        if tied and "lm_head.weight" in loaded_keys:
            errors.append("UNEXPECTED: lm_head.weight present but tie_word_embeddings=true")

    return errors


def main():
    print(f"Reading: {SAFETENSORS_PATH}")
    config = load_config()
    lines = dump_tensors()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tensor dump written to: {OUTPUT_PATH}")
    print(f"Total tensors: {len(lines)}")
    print()

    print("--- Shape assertions ---")
    errors = assert_shapes(config, lines)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print(f"\n{len(errors)} assertion(s) failed.")
        sys.exit(1)
    else:
        print("  All shapes match expected values. PASS")

    print()
    print("--- Tensor dump preview (first 20) ---")
    for line in lines[:20]:
        print(" ", line)
    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more in notes/tensor_dump.txt)")


if __name__ == "__main__":
    main()
