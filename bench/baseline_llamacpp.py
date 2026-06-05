"""
llama.cpp baseline — run on PACE A100 (Phase 3.4).

Runs llama.cpp CLI via subprocess, parses timing output,
writes CSV row with backend="llamacpp".

Usage (PACE only):
    python bench/baseline_llamacpp.py \\
        --gguf-path weights/model.gguf \\
        --llama-cpp-bin llama.cpp/build/bin/llama-cli \\
        --max-tokens 128 --n-runs 3

Prerequisites (do once on PACE):
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON && cmake --build build -j4
    python convert_hf_to_gguf.py ../llm_inference_engine/weights/ --outtype f16 \\
        --outfile ../llm_inference_engine/weights/model.gguf
"""

import argparse
import re
import subprocess
import time
from pathlib import Path


def _parse_timings(stderr: str) -> dict:
    """
    Parse llama_print_timings lines from llama.cpp stderr.
    Example lines:
        llama_print_timings:        load time =   523.45 ms
        llama_print_timings:      prompt eval time =  1234.56 ms /  16 tokens
        llama_print_timings:             eval time =  5678.90 ms / 127 tokens
    """
    result = {}

    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", stderr)
    if m:
        prompt_ms = float(m.group(1))
        n_prompt  = int(m.group(2))
        result["ttft_s"]         = round(prompt_ms / 1000, 6)
        result["n_prompt_tokens"] = n_prompt
        result["prefill_tok_s"]  = round(n_prompt / (prompt_ms / 1000), 3) if prompt_ms > 0 else 0.0

    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", stderr)
    if m:
        eval_ms  = float(m.group(1))
        n_decode = int(m.group(2))
        result["n_decode_tokens"] = n_decode
        result["decode_tok_s"]   = round(n_decode / (eval_ms / 1000), 3) if eval_ms > 0 else 0.0
        result["itl_p50_ms"]     = round(eval_ms / n_decode, 3) if n_decode > 0 else 0.0
        result["itl_p99_ms"]     = result["itl_p50_ms"]  # llama.cpp reports mean, not p99

    return result


def main():
    parser = argparse.ArgumentParser(description="llama.cpp baseline")
    parser.add_argument("--gguf-path",     required=True,  help="Path to .gguf model file")
    parser.add_argument("--llama-cpp-bin", required=True,  help="Path to llama-cli binary")
    parser.add_argument("--max-tokens",    type=int, default=128)
    parser.add_argument("--n-warmup",      type=int, default=1)
    parser.add_argument("--n-runs",        type=int, default=3)
    parser.add_argument("--results-dir",   default="bench/results")
    args = parser.parse_args()

    from bench.harness import PROMPTS, write_results, _hw_metadata

    rows = []

    for prompt_key, prompt_text in PROMPTS.items():
        print(f"\n  [{prompt_key}]", flush=True)

        cmd = [
            args.llama_cpp_bin,
            "-m", args.gguf_path,
            "-p", prompt_text,
            "-n", str(args.max_tokens),
            "--no-display-prompt",
            "-ngl", "99",   # offload all layers to GPU
        ]

        for i in range(args.n_warmup):
            print(f"    warmup {i+1}/{args.n_warmup} ... ", end="", flush=True)
            subprocess.run(cmd, capture_output=True, text=True)
            print("done", flush=True)

        for i in range(args.n_runs):
            print(f"    run {i+1}/{args.n_runs} ... ", end="", flush=True)
            t_start = time.perf_counter()
            proc    = subprocess.run(cmd, capture_output=True, text=True)
            total_s = time.perf_counter() - t_start

            timings = _parse_timings(proc.stderr)
            timings["total_s"] = round(total_s, 6)
            timings.setdefault("peak_mem_mb", 0.0)

            print(
                f"TTFT={timings.get('ttft_s', 0):.2f}s  "
                f"decode={timings.get('decode_tok_s', 0):.1f} tok/s",
                flush=True,
            )
            rows.append({"prompt_key": prompt_key, "run": i + 1, **timings})

    for row in rows:
        row.update({"backend": "llamacpp", **_hw_metadata()})

    write_results(rows, "llamacpp", Path(args.results_dir))


if __name__ == "__main__":
    main()
