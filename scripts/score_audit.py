#!/usr/bin/env python3
"""Score the independent AI-adjudicated audit without exposing model answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = {"short": 0, "medium": 1, "long": 2}
DURATION = {2: "short", 4: "medium", 6: "long"}


def weighted_kappa(first: list[str], second: list[str]) -> float:
    n = len(first)
    observed = sum((ORDER[a] - ORDER[b]) ** 2 / 4 for a, b in zip(first, second)) / n
    frequencies = [
        [values.count(label) / n for label in ORDER]
        for values in (first, second)
    ]
    expected = sum(
        frequencies[0][i] * frequencies[1][j] * ((i - j) ** 2 / 4)
        for i in range(3) for j in range(3)
    )
    return 1 - observed / expected if expected else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = list(map(json.loads, args.expected.read_text(encoding="utf-8").splitlines()))
    judgments = list(map(json.loads, args.judgments.read_text(encoding="utf-8").splitlines()))
    if len(expected) != 200 or len(judgments) != 200:
        raise RuntimeError("audit must contain exactly 200 rows")
    predicted, judged = [], []
    severe_short = unnecessary_long = agreements = 0
    for prediction, judgment in zip(expected, judgments):
        if prediction["id"] != judgment["id"]:
            raise RuntimeError("audit judgment order changed")
        predicted_label = DURATION[prediction["duration"]]
        judged_label = judgment["final_label"]
        predicted.append(predicted_label)
        judged.append(judged_label or ("long" if predicted_label == "short" else "short"))
        agreements += bool(judged_label and predicted_label == judged_label)
        if prediction["audit_slice"] == "accepted_override":
            severe_short += predicted_label == "short" and judged_label == "long"
            unnecessary_long += predicted_label == "long" and judged_label != "long"
    report = {
        "schema": 1,
        "description": "AI-adjudicated, not human-labelled",
        "examples": 200,
        "agreement": agreements / 200,
        "weighted_kappa": weighted_kappa(predicted, judged),
        "severe_short_under_duration_errors": severe_short,
        "unnecessary_long_overrides": unnecessary_long,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if (
        report["agreement"] < 0.90
        or report["weighted_kappa"] < 0.80
        or severe_short
        or unnecessary_long
    ):
        raise RuntimeError("independent audit gate failed")


if __name__ == "__main__":
    main()
