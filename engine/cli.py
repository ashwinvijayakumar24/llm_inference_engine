"""CLI entry point: llm-generate --prompt --max-tokens --temp --top-k --top-p --seed"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Engine — generate text")
    parser.add_argument("--prompt",     required=True,              help="User prompt text")
    parser.add_argument("--max-tokens", type=int,   default=256,    help="Max tokens to generate")
    parser.add_argument("--temp",       type=float, default=1.0,    help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k",      type=int,   default=0,      help="Top-k filter (0=disabled)")
    parser.add_argument("--top-p",      type=float, default=1.0,    help="Top-p nucleus filter (1.0=disabled)")
    parser.add_argument("--seed",       type=int,   default=None,   help="RNG seed for reproducibility")
    parser.add_argument("--weights",    default="weights",          help="Path to weights directory")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from engine.loader import load_config, load_weights
    from engine.model import LlamaModel
    from engine.sampler import get_sampler
    from engine.scheduler import generate

    config  = load_config(args.weights)
    weights = load_weights(args.weights, config)
    model   = LlamaModel(weights, config)

    tokenizer  = AutoTokenizer.from_pretrained(args.weights)
    messages   = [{"role": "user", "content": args.prompt}]
    token_ids  = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    sampler_fn = get_sampler(temp=args.temp, top_k=args.top_k, top_p=args.top_p, seed=args.seed)

    for token_id in generate(model, token_ids, sampler_fn, max_tokens=args.max_tokens):
        print(tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True)
    print()


if __name__ == "__main__":
    main()
