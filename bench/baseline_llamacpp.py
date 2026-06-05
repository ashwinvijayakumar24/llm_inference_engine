"""
llama.cpp baseline — run on PACE A100 (Phase 3.4).

Uses llama-bench (llama.cpp's purpose-built benchmark tool) with CSV output.
llama-bench runs non-interactively and reports prefill (pp) and decode (tg)
tokens/sec directly — no fragile stdout parsing of the chat CLI.

Usage (PACE only):
    python -m bench.baseline_llamacpp \\
        --gguf-path weights/model.gguf \\
        --llama-bench-bin ../llama.cpp/build/bin/llama-bench \\
        --max-tokens 128

Prerequisites (do once on PACE):
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON && cmake --build build -j4
    pip install gguf sentencepiece
    python convert_hf_to_gguf.py ../llm_inference_engine/weights/ --outtype f16 \\
        --outfile ../llm_inference_engine/weights/model.gguf

Note: llama-bench uses synthetic prompts of a fixed token length (not our
short/medium/long text). tok/s depends on length, not content, so this is a
fair throughput comparison. We sweep the same prompt lengths our prompts use.
"""

import argparse
import csv
import io
import subprocess
from pathlib import Path

# Prompt lengths (tokens) roughly matching harness short/medium/long prompts.
PROMPT_LENS = {"short": 40, "medium": 46, "long": 57}


def _run_llama_bench(bin_path: str, gguf: str, n_prompt: int, n_gen: int, ngl: int) -> list[dict]:
    """Run one llama-bench invocation, return parsed CSV rows."""
    cmd = [
        bin_path,
        "-m", gguf,
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-ngl", str(ngl),
        "-o", "csv",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed:\n{proc.stderr}")
    reader = csv.DictReader(io.StringIO(proc.stdout))
    return list(reader)


def main():
    parser = argparse.ArgumentParser(description="llama.cpp baseline via llama-bench")
    parser.add_argument("--gguf-path",       required=True, help="Path to .gguf model file")
    parser.add_argument("--llama-bench-bin", required=True, help="Path to llama-bench binary")
    parser.add_argument("--max-tokens",      type=int, default=128, help="Decode tokens (tg)")
    parser.add_argument("--ngl",             type=int, default=99,  help="GPU layers to offload")
    parser.add_argument("--results-dir",     default="bench/results")
    args = parser.parse_args()

    from bench.harness import write_results, _hw_metadata

    rows = []

    for prompt_key, n_prompt in PROMPT_LENS.items():
        print(f"\n  [{prompt_key}] pp={n_prompt} tg={args.max_tokens} ... ", end="", flush=True)
        bench_rows = _run_llama_bench(
            args.llama_bench_bin, args.gguf_path, n_prompt, args.max_tokens, args.ngl
        )

        prefill_tok_s = 0.0
        decode_tok_s  = 0.0
        for br in bench_rows:
            np_ = int(br.get("n_prompt", 0))
            ng_ = int(br.get("n_gen", 0))
            ts  = float(br.get("avg_ts", 0.0))
            if np_ > 0 and ng_ == 0:
                prefill_tok_s = round(ts, 3)   # pp = prompt processing (prefill)
            elif ng_ > 0 and np_ == 0:
                decode_tok_s = round(ts, 3)    # tg = text generation (decode)

        itl_ms = round(1000.0 / decode_tok_s, 3) if decode_tok_s > 0 else 0.0
        print(f"prefill={prefill_tok_s:.1f} tok/s  decode={decode_tok_s:.1f} tok/s", flush=True)

        rows.append({
            "prompt_key":      prompt_key,
            "run":             1,
            "n_prompt_tokens": n_prompt,
            "n_decode_tokens": args.max_tokens,
            "ttft_s":          0.0,             # llama-bench reports throughput, not TTFT
            "total_s":         0.0,
            "prefill_tok_s":   prefill_tok_s,
            "decode_tok_s":    decode_tok_s,
            "itl_p50_ms":      itl_ms,          # derived from mean decode rate
            "itl_p99_ms":      itl_ms,          # llama-bench reports mean, not p99
            "peak_mem_mb":     0.0,
        })

    for row in rows:
        row.update({"backend": "llamacpp", **_hw_metadata()})

    write_results(rows, "llamacpp", Path(args.results_dir))


if __name__ == "__main__":
    main()
