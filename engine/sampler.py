"""Token sampling: greedy, temperature, top-k, top-p."""

import numpy as np


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng=None,
) -> int:
    """
    Sample a token from logits with optional temperature, top-k, and top-p.

    Pipeline: temperature scale → top-k filter → softmax → top-p filter → sample.
    rng: np.random.Generator (for seeded reproducibility); if None uses global RNG.
    """
    logits = logits.astype(np.float64)

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k: zero out all but the k highest logits
    if top_k > 0:
        k = min(top_k, len(logits))
        topk_idx = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, float("-inf"))
        mask[topk_idx] = logits[topk_idx]
        logits = mask

    # Softmax (numerically stable)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Top-p (nucleus): keep the minimal set whose cumulative prob >= p
    if top_p < 1.0:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        # Keep up to and including the token that pushes cumsum past p
        cutoff      = int(np.searchsorted(cumsum, top_p, side="right")) + 1
        keep        = sorted_idx[:cutoff]
        mask        = np.zeros_like(probs)
        mask[keep]  = probs[keep]
        probs       = mask / mask.sum()

    if rng is not None:
        return int(rng.choice(len(probs), p=probs))
    return int(np.random.choice(len(probs), p=probs))


def get_sampler(
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
):
    """
    Return a callable (logits) -> token_id based on sampling parameters.
    If temp == 0.0, returns greedy (argmax) regardless of other params.
    """
    if temp == 0.0:
        return greedy

    rng = np.random.default_rng(seed) if seed is not None else None

    def _sample(logits: np.ndarray) -> int:
        return sample(logits, temperature=temp, top_k=top_k, top_p=top_p, rng=rng)

    return _sample
