#!/usr/bin/env python3
"""Create deterministic balanced Steady train/test files from teacher labels."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-label", type=int, default=5000)
    parser.add_argument("--test-per-label", type=int, default=1000)
    parser.add_argument(
        "--all-training",
        action="store_true",
        help="write every deduplicated row to train.txt and an empty test.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    seen: dict[str, str] = {}
    for path in args.input:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["label"] = row["label"].split("_", 1)[0]
            prompt_key = " ".join(row["prompt"].lower().split())
            previous = seen.get(prompt_key)
            if previous is not None:
                if previous != row["label"]:
                    raise RuntimeError(f"conflicting labels for duplicate prompt in {path}")
                continue
            seen[prompt_key] = row["label"]
            groups[row["label"]].append(row)

    rng = random.Random(args.seed)
    train: list[str] = []
    test: list[str] = []
    if args.all_training:
        for label, rows in sorted(groups.items()):
            train.extend(f"__label__{label} {row['prompt']}" for row in rows)
        rng.shuffle(train)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "train.txt").write_text("\n".join(train) + "\n")
        (args.output_dir / "test.txt").write_text("")
        print(json.dumps({"train": len(train), "test": 0}))
        return

    required = args.train_per_label + args.test_per_label
    for label in sorted(groups):
        rows = groups[label]
        rng.shuffle(rows)
        if len(rows) < required:
            raise RuntimeError(f"{label} has {len(rows)} rows; requires {required}")
        test.extend(f"__label__{label} {row['prompt']}" for row in rows[: args.test_per_label])
        train.extend(f"__label__{label} {row['prompt']}" for row in rows[args.test_per_label:required])
    rng.shuffle(train)
    rng.shuffle(test)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_text("\n".join(train) + "\n")
    (args.output_dir / "test.txt").write_text("\n".join(test) + "\n")
    print(json.dumps({"train": len(train), "test": len(test)}))


if __name__ == "__main__":
    main()
