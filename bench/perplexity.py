"""
Perplexity eval (Phase 4.1) — fp16 vs int8 vs int4 on a fixed text.

Perplexity = exp(mean(-log p(true_next_token))). Lower is better. It measures how
well the model predicts held-out text; a quantization scheme that barely changes
perplexity has barely changed model quality.

Teacher forcing: run one no-cache forward over the whole token sequence, read the
log-prob the model assigned to each actual next token, average the negative
log-probs, exponentiate.

Usage (PACE):
    python -m bench.perplexity --mode fp16
    python -m bench.perplexity --mode int8
    python -m bench.perplexity --mode int4 --group-size 128
"""

import argparse
from pathlib import Path

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compute_perplexity(model, token_ids: list[int]) -> float:
    """
    token_ids: full sequence. We predict token[i+1] from the logits at position i.
    """
    logits = model.forward_all(token_ids)          # (seq, vocab) fp32 numpy
    logp   = _log_softmax(logits)                  # (seq, vocab)

    # Position i predicts token i+1. Gather log-prob of the true next token.
    nll = []
    for i in range(len(token_ids) - 1):
        true_next = token_ids[i + 1]
        nll.append(-logp[i, true_next])

    mean_nll = float(np.mean(nll))
    return float(np.exp(mean_nll))


def main():
    parser = argparse.ArgumentParser(description="Perplexity eval for fp16/int8/int4")
    parser.add_argument("--mode",       default="fp16", choices=["fp16", "int8", "int4"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--weights",    default="weights")
    parser.add_argument("--text",       default="bench/wikitext_sample.txt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Cap eval length")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from engine.loader import load_config, load_weights_gpu, load_weights_gpu_quant
    from engine.model_gpu import LlamaModelGPU

    text      = Path(args.text).read_text()
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    token_ids = tokenizer.encode(text)[: args.max_tokens]
    print(f"Eval text: {len(token_ids)} tokens", flush=True)

    config = load_config(args.weights)

    print(f"Loading {args.mode} model...", flush=True)
    if args.mode == "fp16":
        weights = load_weights_gpu(args.weights, config)
    else:
        weights = load_weights_gpu_quant(args.weights, config, mode=args.mode,
                                         group_size=args.group_size)
    model = LlamaModelGPU(weights, config)

    ppl = compute_perplexity(model, token_ids)
    print(f"\n  mode={args.mode}  perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()
