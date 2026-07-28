#!/usr/bin/env python3
"""Fetch the fixed 619 FETV prompts for external evaluation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/lyx97/FETV/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
SOURCE_REVISION = "e9a6c057cb6ee9257f29e44d427117e8bd0d704f"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.cache.exists():
        urllib.request.urlretrieve(URL, args.cache)
    table = pq.read_table(args.cache)
    column = next(name for name in ("prompt", "text", "caption") if name in table.column_names)
    prompts = [" ".join(str(value).split()) for value in table[column].to_pylist() if value]
    if len(prompts) != 619:
        raise RuntimeError(f"FETV row count changed: {len(prompts)}")
    body = "".join(
        json.dumps({"id": index, "prompt": prompt}, sort_keys=True) + "\n"
        for index, prompt in enumerate(prompts)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    args.manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "rows": 619,
                "license": "CC-BY-4.0",
                "source_revision": SOURCE_REVISION,
                "usage": "external evaluation only",
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
