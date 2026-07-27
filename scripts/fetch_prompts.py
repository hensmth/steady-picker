#!/usr/bin/env python3
"""Download prompt-only public corpora without downloading media."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


SOURCES = {
    "diffusiondb": (
        "https://huggingface.co/datasets/poloclub/diffusiondb/"
        "resolve/main/metadata.parquet"
    ),
    "fetv": (
        "https://huggingface.co/datasets/lyx97/FETV/resolve/"
        "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    ),
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)


def prompt_column(table) -> str:
    for candidate in ("prompt", "p", "caption", "text"):
        if candidate in table.column_names:
            return candidate
    raise RuntimeError(f"no prompt column in {table.column_names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, default="diffusiondb")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet = args.cache / f"{args.source}.parquet"
    download(SOURCES[args.source], parquet)
    table = pq.read_table(parquet)
    column = prompt_column(table)
    prompts = {
        " ".join(str(value).split())
        for value in table[column].to_pylist()
        if value and 8 <= len(str(value).strip()) <= 2_000
    }
    ordered = sorted(prompts)
    random.Random(args.seed).shuffle(ordered)
    selected = ordered[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(selected):
            handle.write(json.dumps({"id": index, "prompt": prompt}) + "\n")
    print(json.dumps({"source": args.source, "rows": len(selected)}))


if __name__ == "__main__":
    main()
