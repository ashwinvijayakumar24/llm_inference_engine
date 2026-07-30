#!/usr/bin/env python3
"""
Download the WikiText-2 (raw) test split to a plain text file.

Perplexity is only comparable to published numbers when it is measured on a
standard corpus with a standard protocol. This fetches the corpus; run
bench/perplexity.py with --window/--stride for the protocol.

Usage:
    pip install datasets
    python scripts/fetch_wikitext2.py                    # -> bench/wikitext2_test.txt
    python scripts/fetch_wikitext2.py --split validation

Then:
    python -m bench.perplexity --mode fp16 --text bench/wikitext2_test.txt \
        --window 512 --stride 256 --max-tokens 100000
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fetch WikiText-2 raw split as text")
    parser.add_argument("--split",  default="test", choices=["test", "validation", "train"])
    parser.add_argument("--out",    default=None, help="Output path (default: bench/wikitext2_<split>.txt)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The `datasets` package is required.\n"
            "  pip install datasets\n"
            "Or install the bench extra:  pip install -e '.[bench]'"
        )

    out = Path(args.out) if args.out else Path("bench") / f"wikitext2_{args.split}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading wikitext-2-raw-v1 [{args.split}] ...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)

    # Standard treatment: concatenate all lines into one continuous document.
    # The raw split keeps original punctuation and casing (unlike the non-raw
    # version, which is pre-tokenized with <unk> substitutions).
    text = "\n\n".join(ds["text"])
    out.write_text(text)

    print(f"Wrote {out}  ({len(text):,} chars, {len(ds):,} lines)")
    print("\nNext:")
    print(f"  python -m bench.perplexity --mode fp16 --text {out} --window 512 --stride 256 --max-tokens 100000")
    print(f"  python -m bench.perplexity --mode int8 --text {out} --window 512 --stride 256 --max-tokens 100000")
    print(f"  python -m bench.perplexity --mode int4 --text {out} --window 512 --stride 256 --max-tokens 100000")


if __name__ == "__main__":
    main()
