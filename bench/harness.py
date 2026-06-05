#!/usr/bin/env python3
"""
Benchmark harness: TTFT, decode tok/s, p50/p99 inter-token latency.

Usage (Mac CPU):
    python bench/harness.py --max-tokens 128 --n-runs 3

Usage (PACE A100 — after model_gpu.py is implemented):
    python bench/harness.py --backend gpu --max-tokens 128 --n-runs 5

Output: bench/results/<timestamp>_<host>_<backend>.{json,csv}
"""

import argparse
import csv
import json
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Prompt set — short / medium / long
# ---------------------------------------------------------------------------

PROMPTS = {
    "short": "Hello, I am",
    "medium": "The quick brown fox jumped over the lazy dog.",
    "long": (
        "Explain the fundamental theorem of calculus in detail, "
        "including both the first and second parts, with examples."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_mem_mb() -> float:
    """Peak RSS in MB. macOS: bytes; Linux: kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def _hw_metadata() -> dict:
    return {
        "hostname":   platform.node(),
        "platform":   platform.platform(),
        "python_ver": platform.python_version(),
        "numpy_ver":  np.__version__,
    }


# ---------------------------------------------------------------------------
# Core timing function
# ---------------------------------------------------------------------------

def time_generate(model, token_ids: list[int], sampler_fn, max_tokens: int) -> dict:
    """
    Run one generate() call with external timing probes.
    No changes to generate() — all measurement is here.

    Returns dict with TTFT, ITL percentiles, tok/s, peak memory.
    """
    from engine.scheduler import generate

    timestamps: list[float] = []
    t_start = time.perf_counter()

    for _ in generate(model, token_ids, sampler_fn, max_tokens=max_tokens):
        timestamps.append(time.perf_counter())

    if not timestamps:
        return {}

    n_decode  = len(timestamps)
    ttft      = timestamps[0] - t_start
    itls      = [timestamps[i] - timestamps[i - 1] for i in range(1, n_decode)]
    total_s   = timestamps[-1] - t_start

    return {
        "n_prompt_tokens": len(token_ids),
        "n_decode_tokens": n_decode,
        "ttft_s":          round(ttft, 6),
        "total_s":         round(total_s, 6),
        "prefill_tok_s":   round(len(token_ids) / ttft, 3) if ttft > 0 else 0.0,
        "decode_tok_s":    round((n_decode - 1) / sum(itls), 3) if itls else 0.0,
        "itl_p50_ms":      round(_percentile(itls, 50) * 1000, 3),
        "itl_p99_ms":      round(_percentile(itls, 99) * 1000, 3),
        "peak_mem_mb":     round(_peak_mem_mb(), 1),
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    model,
    tokenizer,
    sampler_fn,
    max_tokens: int,
    n_warmup: int,
    n_runs: int,
) -> list[dict]:
    rows = []

    for prompt_key, prompt_text in PROMPTS.items():
        messages   = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        token_ids  = tokenizer.encode(prompt_str)

        print(f"\n  [{prompt_key}] {len(token_ids)} prompt tokens", flush=True)

        for i in range(n_warmup):
            print(f"    warmup {i + 1}/{n_warmup} ... ", end="", flush=True)
            time_generate(model, token_ids, sampler_fn, max_tokens)
            print("done", flush=True)

        for i in range(n_runs):
            print(f"    run {i + 1}/{n_runs} ... ", end="", flush=True)
            result = time_generate(model, token_ids, sampler_fn, max_tokens)
            print(
                f"TTFT={result['ttft_s']:.2f}s  "
                f"decode={result['decode_tok_s']:.1f} tok/s  "
                f"p50ITL={result['itl_p50_ms']:.1f}ms  "
                f"p99ITL={result['itl_p99_ms']:.1f}ms",
                flush=True,
            )
            rows.append({"prompt_key": prompt_key, "run": i + 1, **result})

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_results(rows: list[dict], backend: str, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{platform.node()}_{backend}"

    json_path = results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    csv_path = results_dir / f"{stem}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nResults:\n  {json_path}\n  {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM inference benchmark harness")
    parser.add_argument("--backend",     default="cpu",          help="cpu | gpu")
    parser.add_argument("--max-tokens",  type=int, default=128,  help="Decode tokens per run")
    parser.add_argument("--n-warmup",    type=int, default=1,    help="Warmup runs (discarded)")
    parser.add_argument("--n-runs",      type=int, default=3,    help="Measured runs per prompt")
    parser.add_argument("--weights",     default="weights",       help="Path to weights directory")
    parser.add_argument("--results-dir", default="bench/results", help="Output directory")
    args = parser.parse_args()

    print("Loading model...", flush=True)

    from transformers import AutoTokenizer
    from engine.loader import load_config
    from engine.sampler import greedy

    config    = load_config(args.weights)
    tokenizer = AutoTokenizer.from_pretrained(args.weights)

    if args.backend == "gpu":
        from engine.loader import load_weights_gpu
        from engine.model_gpu import LlamaModelGPU
        weights = load_weights_gpu(args.weights, config)
        model   = LlamaModelGPU(weights, config)
    else:
        from engine.loader import load_weights
        from engine.model import LlamaModel
        weights = load_weights(args.weights, config)
        model   = LlamaModel(weights, config)

    meta = _hw_metadata()
    print(f"Hardware: {meta['platform']}")
    print(f"Backend: {args.backend}  max_tokens={args.max_tokens}  "
          f"warmup={args.n_warmup}  runs={args.n_runs}")

    rows = run_benchmark(model, tokenizer, greedy, args.max_tokens, args.n_warmup, args.n_runs)

    for row in rows:
        row.update({"backend": args.backend, **meta})

    write_results(rows, args.backend, Path(args.results_dir))


if __name__ == "__main__":
    main()
