"""
HF oracle — runs Llama 3.2 1B in fp32 and captures intermediate states.

Run once to generate fixtures:
    python3 tests/oracle.py

Fixtures saved to tests/fixtures/oracle_{key}.pkl.
All downstream tests load from these pickles — never re-runs HF unless you delete fixtures.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixed prompts — do not change; would invalidate saved fixtures
PROMPTS = {
    "short": "Hello, I am",
    "medium": "The quick brown fox jumped over the lazy dog.",
}

EOS_IDS = {128001, 128008, 128009}
GREEDY_TOKENS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fixture_path(key: str) -> Path:
    return FIXTURES_DIR / f"oracle_{key}.pkl"


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    name: str,
    atol: float = 1e-4,
    rtol: float = 0.0,
) -> bool:
    """Compare two arrays. Print stats. Return True if within tolerance."""
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    idx = np.unravel_index(diff.argmax(), diff.shape)
    passed = max_diff <= atol
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {name:50s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
        f"  worst_idx={idx}  atol={atol:.0e}"
    )
    return passed


# ---------------------------------------------------------------------------
# Oracle generation
# ---------------------------------------------------------------------------

def _load_hf_model():
    """Load HF model in fp32, eval mode."""
    print("Loading HF model (fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS_DIR),
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    return model


def _load_tokenizer():
    return AutoTokenizer.from_pretrained(str(WEIGHTS_DIR))


def generate_fixture(key: str, prompt: str, model, tokenizer) -> dict:
    """
    Run one forward pass + greedy decode on prompt.
    Returns a dict of captured states (numpy fp32 arrays).
    Saves to tests/fixtures/oracle_{key}.pkl.
    """
    path = fixture_path(key)
    if path.exists():
        print(f"  Fixture '{key}' already exists — loading from disk.")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Generating fixture '{key}' for prompt: {repr(prompt)}")

    tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tok_ids], dtype=torch.long)  # (1, seq)

    captures = {}

    # --- Hook registration ---
    hooks = []

    # post-embed
    def make_embed_hook():
        def hook(module, inp, out):
            captures["post_embed"] = out[0].detach().float().numpy()  # (seq, hidden)
        return hook
    hooks.append(
        model.model.embed_tokens.register_forward_hook(make_embed_hook())
    )

    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        layer = model.model.layers[i]

        # post-attn + residual 1: input to post_attention_layernorm
        def make_post_attn_hook(idx):
            def hook(module, inp, out):
                # inp[0] is the hidden state before this norm = post-attn + residual
                captures[f"layer_{idx}_post_attn"] = inp[0].detach().float().numpy()
            return hook
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(make_post_attn_hook(i))
        )

        # post-FFN + residual 2: output of the full layer
        def make_post_ffn_hook(idx):
            def hook(module, inp, out):
                # out is a tuple; [0] is hidden states
                h = out[0] if isinstance(out, tuple) else out
                captures[f"layer_{idx}_post_ffn"] = h.detach().float().numpy()
            return hook
        hooks.append(
            layer.register_forward_hook(make_post_ffn_hook(i))
        )

    # post-final-norm
    def make_final_norm_hook():
        def hook(module, inp, out):
            captures["post_final_norm"] = out.detach().float().numpy()
        return hook
    hooks.append(
        model.model.norm.register_forward_hook(make_final_norm_hook())
    )

    # logits: output of lm_head
    def make_logits_hook():
        def hook(module, inp, out):
            captures["logits"] = out.detach().float().numpy()
        return hook
    hooks.append(
        model.lm_head.register_forward_hook(make_logits_hook())
    )

    # --- Forward pass ---
    with torch.no_grad():
        model(input_ids)

    # --- Remove hooks ---
    for h in hooks:
        h.remove()

    # Squeeze batch dim on all captures
    for k, v in captures.items():
        if v.ndim == 3:          # (1, seq, hidden) → (seq, hidden)
            captures[k] = v[0]
        elif v.ndim == 2 and v.shape[0] == 1:  # (1, vocab) — unlikely but safe
            captures[k] = v[0]

    captures["token_ids"] = tok_ids

    # --- Greedy decode (32 tokens) ---
    print(f"    Running greedy decode ({GREEDY_TOKENS} tokens)...")
    generated_ids = list(tok_ids)
    with torch.no_grad():
        for _ in range(GREEDY_TOKENS):
            inp = torch.tensor([generated_ids], dtype=torch.long)
            out = model(inp)
            next_id = int(out.logits[0, -1].argmax())
            generated_ids.append(next_id)
            if next_id in EOS_IDS:
                break
    captures["greedy_ids"] = generated_ids[len(tok_ids):]

    # --- Save ---
    FIXTURES_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(captures, f)
    print(f"    Saved to {path}")
    print(f"    Keys: {sorted(captures.keys())[:6]} ...")

    return captures


def load_fixture(key: str) -> dict:
    path = fixture_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture '{key}' missing. Run: python3 tests/oracle.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main — generate all fixtures
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = _load_hf_model()
    tokenizer = _load_tokenizer()

    print("\n=== Generating oracle fixtures ===")
    for key, prompt in PROMPTS.items():
        fx = generate_fixture(key, prompt, model, tokenizer)

    print("\n=== Verifying determinism (re-run short fixture) ===")
    # Regenerate short fixture from scratch and compare
    key = "short"
    prompt = PROMPTS[key]
    tok_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([tok_ids], dtype=torch.long)
    with torch.no_grad():
        out1 = model(input_ids).logits[0].float().numpy()
        out2 = model(input_ids).logits[0].float().numpy()
    diff = float(np.abs(out1 - out2).max())
    assert diff == 0.0, f"Non-deterministic! max diff = {diff}"
    print(f"  [PASS] logits identical across two runs (max diff = {diff})")

    print("\nAll fixtures generated.")
