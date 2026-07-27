#!/usr/bin/env python3
"""Create deterministic stratified cross-validation folds."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    groups: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[str] = set()
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        prompt = " ".join(row["prompt"].split())
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        groups[row["label"].split("_", 1)[0]].append(prompt)

    rng = random.Random(args.seed)
    for prompts in groups.values():
        rng.shuffle(prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.folds):
        train: list[str] = []
        test: list[str] = []
        for label, prompts in sorted(groups.items()):
            for index, prompt in enumerate(prompts):
                record = f"__label__{label} {prompt}"
                (test if index % args.folds == fold else train).append(record)
        random.Random(args.seed + fold).shuffle(train)
        random.Random(args.seed + args.folds + fold).shuffle(test)
        fold_dir = args.output_dir / f"fold-{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train.txt").write_text("\n".join(train) + "\n")
        (fold_dir / "test.txt").write_text("\n".join(test) + "\n")
        print(json.dumps({"fold": fold + 1, "train": len(train), "test": len(test)}))


if __name__ == "__main__":
    main()
