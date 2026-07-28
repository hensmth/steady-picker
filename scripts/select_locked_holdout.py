#!/usr/bin/env python3
"""Select a deterministic, source-balanced temporal-complexity holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


QUOTAS = {
    ("simple", "VideoUFO"): 250,
    ("simple", "DiffusionDB"): 50,
    ("moderate", "VideoUFO"): 250,
    ("moderate", "DiffusionDB"): 50,
    ("high", "VideoUFO"): 100,
    ("high", "DiffusionDB"): 300,
}


def band(score: int) -> str:
    if score <= 0:
        return "simple"
    if score <= 2:
        return "moderate"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    groups: dict[tuple[str, str], list[dict]] = {key: [] for key in QUOTAS}
    for row in rows:
        key = (band(int(row["temporal_complexity"])), row["source"])
        if key in groups:
            groups[key].append(row)
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for key, quota in QUOTAS.items():
        candidates = groups[key]
        rng.shuffle(candidates)
        if len(candidates) < quota:
            raise RuntimeError(f"{key} has {len(candidates)} rows; need {quota}")
        selected.extend(candidates[:quota])
    rng.shuffle(selected)
    output = "".join(
        json.dumps(
            {
                **row,
                "id": index,
                "holdout_complexity_band": band(int(row["temporal_complexity"])),
            },
            sort_keys=True,
        )
        + "\n"
        for index, row in enumerate(selected)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "rows": len(selected),
        "quotas": {f"{key[0]}:{key[1]}": value for key, value in QUOTAS.items()},
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
