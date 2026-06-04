"""CLI entry point: llm-generate --prompt --max-tokens --temp --top-k --top-p --seed"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Engine — generate text")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    raise NotImplementedError("CLI not yet implemented — complete Phase 2 first")


if __name__ == "__main__":
    main()
