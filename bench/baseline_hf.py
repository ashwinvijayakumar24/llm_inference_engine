"""
HuggingFace transformers baseline — run on PACE A100 (Phase 3.3).

Loads Llama 3.2 1B via AutoModelForCausalLM in fp16 on cuda:0,
runs same prompts as harness.py, writes CSV row with backend="hf_transformers".

Usage (PACE only):
    python bench/baseline_hf.py --max-tokens 128 --n-runs 3 --weights weights
"""

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="HF transformers baseline")
    parser.add_argument("--max-tokens",  type=int, default=128)
    parser.add_argument("--n-warmup",    type=int, default=1)
    parser.add_argument("--n-runs",      type=int, default=3)
    parser.add_argument("--weights",     default="weights")
    parser.add_argument("--results-dir", default="bench/results")
    args = parser.parse_args()

    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bench.harness import PROMPTS, write_results, _hw_metadata, _percentile

    print("Loading HF model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.weights)
    model     = AutoModelForCausalLM.from_pretrained(
        args.weights, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    print("Model loaded.", flush=True)

    rows = []

    for prompt_key, prompt_text in PROMPTS.items():
        messages  = [{"role": "user", "content": prompt_text}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda:0")

        n_prompt = input_ids.shape[-1]
        print(f"\n  [{prompt_key}] {n_prompt} prompt tokens", flush=True)

        for i in range(args.n_warmup):
            print(f"    warmup {i+1}/{args.n_warmup} ... ", end="", flush=True)
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=args.max_tokens, do_sample=False)
            print("done", flush=True)

        for i in range(args.n_runs):
            print(f"    run {i+1}/{args.n_runs} ... ", end="", flush=True)
            timestamps: list[float] = []
            t_start = time.perf_counter()

            # Stream tokens one at a time for accurate per-token timing
            past_key_values = None
            cur_ids = input_ids
            with torch.no_grad():
                for _ in range(args.max_tokens):
                    out = model(cur_ids, past_key_values=past_key_values, use_cache=True)
                    past_key_values = out.past_key_values
                    next_id = int(out.logits[0, -1].argmax())
                    timestamps.append(time.perf_counter())
                    cur_ids = torch.tensor([[next_id]], device="cuda:0")
                    if next_id in {128001, 128008, 128009}:
                        break

            n_decode = len(timestamps)
            ttft     = timestamps[0] - t_start
            itls     = [timestamps[j] - timestamps[j-1] for j in range(1, n_decode)]
            total_s  = timestamps[-1] - t_start

            result = {
                "n_prompt_tokens": n_prompt,
                "n_decode_tokens": n_decode,
                "ttft_s":          round(ttft, 6),
                "total_s":         round(total_s, 6),
                "prefill_tok_s":   round(n_prompt / ttft, 3) if ttft > 0 else 0.0,
                "decode_tok_s":    round((n_decode - 1) / sum(itls), 3) if itls else 0.0,
                "itl_p50_ms":      round(_percentile(itls, 50) * 1000, 3),
                "itl_p99_ms":      round(_percentile(itls, 99) * 1000, 3),
                "peak_mem_mb":     round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            }
            print(
                f"TTFT={result['ttft_s']:.2f}s  "
                f"decode={result['decode_tok_s']:.1f} tok/s  "
                f"p99ITL={result['itl_p99_ms']:.1f}ms",
                flush=True,
            )
            rows.append({"prompt_key": prompt_key, "run": i + 1, **result})

    for row in rows:
        row.update({"backend": "hf_transformers", **_hw_metadata()})

    write_results(rows, "hf_transformers", Path(args.results_dir))


if __name__ == "__main__":
    main()
