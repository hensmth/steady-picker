#!/usr/bin/env python3
"""Validate sealed holdout integrity/support without revealing labels or counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [
        json.loads(line)
        for line in args.prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        json.loads(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    integrity = len(prompts) == len(labels) == 1_000
    if integrity:
        for expected, labelled in zip(prompts, labels, strict=True):
            if (
                expected["id"] != labelled["id"]
                or expected["prompt"] != labelled["prompt"]
                or not isinstance(labelled.get("structured_votes"), list)
                or len(labelled["structured_votes"]) != 3
            ):
                integrity = False
                break
    short = sum(
        row.get("final_label") == "short" and not row.get("unresolved", True)
        for row in labels
    )
    long = sum(
        row.get("final_label") == "long" and not row.get("unresolved", True)
        for row in labels
    )
    sufficient = integrity and short >= 100 and long >= 100
    result = {
        "schema": 1,
        "row_integrity": integrity,
        "unanimous_override_support_sufficient": sufficient,
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "label_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    if not sufficient:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
