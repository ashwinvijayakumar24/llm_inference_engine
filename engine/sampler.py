"""Token sampling: greedy, temperature, top-k, top-p."""

import numpy as np


def greedy(logits: np.ndarray) -> int:
    raise NotImplementedError


def sample(logits: np.ndarray, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0, rng=None) -> int:
    raise NotImplementedError
