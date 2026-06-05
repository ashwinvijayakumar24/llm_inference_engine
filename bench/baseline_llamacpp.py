"""
llama.cpp baseline — run on PACE A100 (Phase 3.4).

Builds llama.cpp with LLAMA_CUBLAS=1, converts weights to GGUF,
runs same prompts via subprocess, parses timing output,
writes CSV row with backend="llamacpp".

Usage (PACE only):
    python bench/baseline_llamacpp.py --max-tokens 128 --n-runs 5

Prerequisites (do once on PACE):
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make LLAMA_CUBLAS=1
    python convert_hf_to_gguf.py <weights_path> --outtype q8_0
"""


def main():
    raise NotImplementedError(
        "llama.cpp baseline not yet implemented. Run on PACE A100 after building llama.cpp."
    )


if __name__ == "__main__":
    main()
