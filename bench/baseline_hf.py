"""
HuggingFace transformers baseline — run on PACE A100 (Phase 3.3).

Loads Llama 3.2 1B via AutoModelForCausalLM in fp16 on cuda:0,
runs same prompts as harness.py, writes CSV row with backend="hf_transformers".

Usage (PACE only):
    python bench/baseline_hf.py --max-tokens 128 --n-runs 5
"""


def main():
    raise NotImplementedError(
        "HF baseline not yet implemented. Run on PACE A100 after model_gpu.py is complete."
    )


if __name__ == "__main__":
    main()
