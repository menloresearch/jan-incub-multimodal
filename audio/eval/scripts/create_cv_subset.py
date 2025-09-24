#!/usr/bin/env python3
"""Subsample Common Voice TSV splits while preserving format."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a subsampled Common Voice split TSV")
    parser.add_argument(
        "--root",
        default=os.getenv("CV22_PATH"),
        help="Path to Common Voice dataset root (default: env CV22_PATH)",
    )
    parser.add_argument("--lang", default="en", help="Language code to process (default: %(default)s)")
    parser.add_argument("--split", default="test", help="Split name to subsample (default: %(default)s)")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to keep (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: %(default)s)")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename (default: <split>_<n>.tsv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.root is None:
        print("Dataset root not provided and CV22_PATH env not set", file=sys.stderr)
        return 1

    root = Path(args.root).expanduser().resolve()
    lang_dir = root / args.lang
    split_path = lang_dir / f"{args.split}.tsv"

    if not split_path.exists():
        print(f"Split TSV not found: {split_path}", file=sys.stderr)
        return 1

    output_name = args.output_name or f"{args.split}_{args.n}.tsv"
    output_path = lang_dir / output_name

    df = pd.read_csv(split_path, sep="\t", low_memory=False)

    if df.empty:
        print(f"Split {split_path} is empty; nothing to sample", file=sys.stderr)
        return 1

    sample_size = min(args.n, len(df))
    if sample_size < args.n:
        print(f"Warning: requested {args.n} rows but only {sample_size} available")

    sample_df = df.sample(n=sample_size, random_state=args.seed)
    sample_df.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote subset with {sample_size} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
