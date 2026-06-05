"""
Tokenizer round-trip and chat template validation.
Phase 0.3 acceptance criteria — run and must exit 0.

Usage:
    python3 scripts/tokenizer_check.py
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
TOKENIZER_CONFIG_PATH = WEIGHTS_DIR / "tokenizer_config.json"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))


def test_round_trip(tok):
    text = "Hello, world!"
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert decoded == text, f"Round-trip failed: {repr(text)} -> {ids} -> {repr(decoded)}"
    print(f"  [PASS] round-trip: {repr(text)} -> {ids} -> {repr(decoded)}")


def test_special_tokens(tok):
    with open(TOKENIZER_CONFIG_PATH) as f:
        tc = json.load(f)

    expected_bos = 128000
    expected_eos_set = {128001, 128008, 128009}

    assert tok.bos_token_id == expected_bos, (
        f"BOS mismatch: got {tok.bos_token_id}, expected {expected_bos}"
    )
    # eos_token_id may be int or list
    actual_eos = tok.eos_token_id
    if isinstance(actual_eos, int):
        actual_eos_set = {actual_eos}
    else:
        actual_eos_set = set(actual_eos)
    assert actual_eos_set & expected_eos_set, (
        f"EOS mismatch: got {actual_eos_set}, expected subset of {expected_eos_set}"
    )

    print(f"  [PASS] BOS token id: {tok.bos_token_id}  ({tok.bos_token})")
    print(f"  [PASS] EOS token id: {tok.eos_token_id}  ({tok.eos_token})")
    print(f"  [PASS] pad token id: {tok.pad_token_id}  ({tok.pad_token})")


def test_chat_template(tok):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    # apply_chat_template with tokenize=True returns BatchEncoding on fast tokenizers
    result = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    # Extract list[int] regardless of whether result is list or BatchEncoding
    if hasattr(result, "input_ids"):
        ids_ref = result.input_ids
    else:
        ids_ref = list(result)

    # Our usage path: apply_chat_template → string → encode
    # (engine will call apply_chat_template(tokenize=False) then tok.encode)
    text = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    ids_ours = tok.encode(text, add_special_tokens=False)

    assert ids_ref == ids_ours, (
        f"Chat template token mismatch.\n"
        f"  ref:   {ids_ref[:10]}...\n"
        f"  ours:  {ids_ours[:10]}..."
    )

    print(f"  [PASS] chat template: {len(ids_ref)} tokens")
    print(f"         text preview:  {repr(text[:120])}")
    print(f"         first 10 ids:  {ids_ref[:10]}")
    print(f"         last  10 ids:  {ids_ref[-10:]}")

    return text, ids_ref


def main():
    print("Loading tokenizer...")
    tok = load_tokenizer()
    print(f"  vocab size:     {tok.vocab_size}")
    print(f"  model max len:  {tok.model_max_length}")
    print()

    print("--- 0.3.1  Round-trip test ---")
    test_round_trip(tok)
    print()

    print("--- 0.3.2  Special token IDs ---")
    test_special_tokens(tok)
    print()

    print("--- 0.3.3  Chat template ---")
    chat_text, chat_ids = test_chat_template(tok)
    print()

    print("All tokenizer checks passed.")


if __name__ == "__main__":
    main()
