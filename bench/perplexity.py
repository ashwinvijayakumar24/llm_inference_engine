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
    Single-window perplexity: one no-cache forward over the whole sequence.

    token_ids: full sequence. We predict token[i+1] from the logits at position i.
    Only valid when the sequence fits comfortably in context; for longer texts use
    compute_perplexity_strided(), which is the standard protocol.
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


def compute_perplexity_strided(
    model,
    token_ids: list[int],
    window: int = 512,
    stride: int = 256,
    verbose: bool = True,
) -> float:
    """
    Standard sliding-window perplexity, matching the usual WikiText protocol.

    A window of `window` tokens slides forward by `stride` each step. Within each
    window, only the tokens NOT already scored by a previous window contribute to
    the NLL — the overlapping prefix serves purely as context. That is the point of
    the protocol: every scored token gets `window - stride` tokens of left context
    instead of being penalised for sitting at the start of a chunk.

    stride == window degenerates to disjoint chunks (each token gets little or no
    context, inflating perplexity). Smaller stride = more context per token = lower
    perplexity and more compute. Report both numbers; they are not comparable across
    different (window, stride) settings.
    """
    n = len(token_ids)
    if n <= window:
        return compute_perplexity(model, token_ids)

    if stride >= window:
        # With no overlap, the first token of every chunk has no in-chunk
        # predecessor to be predicted from, so it is never scored. Coverage
        # silently drops and the result is not comparable to an overlapped run.
        print(f"  WARNING: stride ({stride}) >= window ({window}) — chunks do not "
              f"overlap, so ~{n // stride} tokens will go unscored and each scored "
              f"token gets little left context. Use stride < window.", flush=True)

    nll: list[float] = []
    prev_end = 0
    n_windows = 0

    for begin in range(0, n, stride):
        end = min(begin + window, n)
        chunk = token_ids[begin:end]

        logits = model.forward_all(chunk)          # (chunk_len, vocab)
        logp   = _log_softmax(logits)

        # Global target index t is predicted by the logits at global position t-1,
        # i.e. chunk row (t - 1 - begin). Score only targets not already counted.
        first_target = max(prev_end, begin + 1)
        for t in range(first_target, end):
            nll.append(-logp[t - 1 - begin, token_ids[t]])

        n_windows += 1
        if verbose:
            print(f"    window {n_windows}: tokens[{begin}:{end}], "
                  f"scored {end - first_target}, running ppl="
                  f"{np.exp(np.mean(nll)):.4f}", flush=True)

        prev_end = end
        if end == n:
            break

    if verbose:
        print(f"  {n_windows} windows, {len(nll)} tokens scored "
              f"(of {n} total)", flush=True)

    return float(np.exp(float(np.mean(nll))))


def main():
    parser = argparse.ArgumentParser(description="Perplexity eval for fp16/int8/int4")
    parser.add_argument("--mode",       default="fp16", choices=["fp16", "int8", "int4"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--weights",    default="weights")
    parser.add_argument("--text",       default="bench/computing_history.txt",
                        help="Eval text. Default is a short synthetic sample — "
                             "deltas between modes are meaningful, the absolute "
                             "perplexity is not comparable to published numbers.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Cap eval length")
    parser.add_argument("--window",     type=int, default=0,
                        help="Sliding-window size. 0 = single-pass over the whole text "
                             "(only sensible for short texts). Use 512 for WikiText-2.")
    parser.add_argument("--stride",     type=int, default=256,
                        help="Sliding-window stride. Ignored unless --window is set.")
    parser.add_argument("--results-dir", default="bench/results",
                        help="Where to append the CSV row.")
    parser.add_argument("--no-csv",     action="store_true", help="Skip writing the CSV row.")
    args = parser.parse_args()

    import csv
    from transformers import AutoTokenizer
    from engine.loader import load_config, load_weights_gpu, load_weights_gpu_quant
    from engine.model_gpu import LlamaModelGPU
    from bench.harness import _hw_metadata, _weight_mem_mb

    text      = Path(args.text).read_text()
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    token_ids = tokenizer.encode(text)[: args.max_tokens]
    protocol  = (f"sliding window={args.window} stride={args.stride}"
                 if args.window else "single pass (no window)")
    print(f"Eval text: {Path(args.text).name}, {len(token_ids)} tokens, {protocol}", flush=True)

    config = load_config(args.weights)

    print(f"Loading {args.mode} model...", flush=True)
    if args.mode == "fp16":
        weights = load_weights_gpu(args.weights, config)
    else:
        weights = load_weights_gpu_quant(args.weights, config, mode=args.mode,
                                         group_size=args.group_size)
    model = LlamaModelGPU(weights, config)
    weight_mem_mb = round(_weight_mem_mb(weights), 1)

    if args.window:
        ppl = compute_perplexity_strided(model, token_ids,
                                         window=args.window, stride=args.stride)
    else:
        ppl = compute_perplexity(model, token_ids)

    print(f"\n  mode={args.mode}  weight_mem={weight_mem_mb} MB  perplexity={ppl:.4f}")

    if args.no_csv:
        return

    # Append one row per run so fp16/int8/int4 accumulate into a single file.
    row = {
        "mode":          args.mode,
        "group_size":    args.group_size if args.mode == "int4" else "",
        "text":          Path(args.text).name,
        "n_tokens":      len(token_ids),
        "window":        args.window or "",
        "stride":        args.stride if args.window else "",
        "perplexity":    round(ppl, 4),
        "weight_mem_mb": weight_mem_mb,
        **_hw_metadata(),
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "perplexity.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            wr.writeheader()
        wr.writerow(row)
    print(f"  Appended to {csv_path}")


if __name__ == "__main__":
    main()
