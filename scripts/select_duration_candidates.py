#!/usr/bin/env python3
"""Select a balanced candidate pool before teacher labelling."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


LONG = re.compile(
    r"\b(?:then|timelapse|time-lapse|transformation|transforming|morph|"
    r"grows?|changes?|sequence|before and after|transition|progression)\b",
    re.IGNORECASE,
)
SHORT = re.compile(
    r"\b(?:quick|brief|blink|wink|jump|flash|burst|single shot|still|portrait|"
    r"close-up|closeup|headshot|logo|icon)\b",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = {"short": [], "normal": [], "long": []}
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        prompt = row["prompt"]
        if LONG.search(prompt):
            groups["long"].append(prompt)
        elif SHORT.search(prompt):
            groups["short"].append(prompt)
        else:
            groups["normal"].append(prompt)

    rng = random.Random(args.seed)
    selected: list[str] = []
    for name in ("short", "normal", "long"):
        rng.shuffle(groups[name])
        if len(groups[name]) < args.per_group:
            raise RuntimeError(f"{name} has only {len(groups[name])} candidates")
        selected.extend(groups[name][: args.per_group])
    rng.shuffle(selected)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(selected):
            handle.write(json.dumps({"id": index, "prompt": prompt}) + "\n")
    print(json.dumps({name: args.per_group for name in groups}))


if __name__ == "__main__":
    main()
