#!/usr/bin/env python3
"""Export resolved teacher JSONL to the classifier's labelled text format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(map(json.loads, args.input.read_text(encoding="utf-8").splitlines()))
    unresolved = [row["id"] for row in rows if row["unresolved"]]
    if unresolved:
        raise RuntimeError(f"cannot export unresolved ids: {unresolved[:10]}")
    args.output.write_text(
        "".join(f"__label__{row['final_label']} {row['prompt']}\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
