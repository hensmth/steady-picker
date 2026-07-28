#!/usr/bin/env python3
"""Run the fixed settings-v2 hyperparameter grid and select one candidate."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
from pathlib import Path


def rows(path: Path) -> list[tuple[str, str]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label, prompt = line.split(" ", 1)
        output.append((label.removeprefix("__label__"), prompt))
    return output


def evaluate(binary: Path, model: Path, validation: list[tuple[str, str]]) -> dict:
    requests = "".join(
        json.dumps({"prompt": prompt, "mode": "text-to-video"}) + "\n"
        for _, prompt in validation
    )
    result = subprocess.run(
        [str(binary), "predict", "--model", str(model)],
        input=requests, text=True, capture_output=True, check=True,
    )
    predictions = list(map(json.loads, result.stdout.splitlines()))
    accepted = {"short": [0, 0], "long": [0, 0]}
    utility = 0.0
    for (actual, _), prediction in zip(validation, predictions):
        if prediction["duration_source"] != "model":
            continue
        predicted = "short" if prediction["duration"] == 2 else "long"
        accepted[predicted][0] += 1
        accepted[predicted][1] += actual == predicted
        utility += 1 if actual == predicted else (-4 if predicted == "short" else -2)
    return {
        "utility": utility / len(validation),
        "short_precision": accepted["short"][1] / accepted["short"][0] if accepted["short"][0] else 0,
        "long_precision": accepted["long"][1] / accepted["long"][0] if accepted["long"][0] else 0,
        "coverage": sum(value[0] for value in accepted.values()) / len(validation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidates = itertools.product(
        (10_000, 20_000), (64, 96), ((3, 5), (3, 6)),
        (0.05, 0.1), (25, 40),
    )
    report = []
    for index, (bucket, dimension, ngrams, learning_rate, epochs) in enumerate(candidates):
        fold_metrics = []
        sizes = []
        for fold in range(1, 6):
            directory = args.folds_dir / f"fold-{fold}"
            artifact = args.work_dir / f"candidate-{index:02d}-fold-{fold}.bin"
            command = [
                str(args.binary), "train",
                "--train", str(directory / "fit.txt"),
                "--probability-calibration", str(directory / "probability.txt"),
                "--conformal-calibration", str(directory / "conformal.txt"),
                "--threshold-development", str(directory / "development.txt"),
                "--output", str(artifact), "--bucket", str(bucket),
                "--dimension", str(dimension), "--min-ngram", str(ngrams[0]),
                "--max-ngram", str(ngrams[1]), "--learning-rate", str(learning_rate),
                "--epochs", str(epochs),
                "--source-manifest-sha256", args.source_manifest_sha256,
                "--training-code-commit", args.training_code_commit,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            fold_metrics.append(evaluate(args.binary, artifact, rows(directory / "validation.txt")))
            sizes.append(artifact.stat().st_size)
            artifact.unlink()
        utilities = [metric["utility"] for metric in fold_metrics]
        candidate = {
            "bucket": bucket, "dimension": dimension, "min_ngram": ngrams[0],
            "max_ngram": ngrams[1], "learning_rate": learning_rate, "epochs": epochs,
            "mean_utility": statistics.mean(utilities),
            "utility_variance": statistics.pvariance(utilities),
            "mean_short_precision": statistics.mean(metric["short_precision"] for metric in fold_metrics),
            "mean_long_precision": statistics.mean(metric["long_precision"] for metric in fold_metrics),
            "mean_coverage": statistics.mean(metric["coverage"] for metric in fold_metrics),
            "mean_artifact_bytes": statistics.mean(sizes),
            "folds": fold_metrics,
        }
        candidate["eligible"] = (
            candidate["mean_short_precision"] >= 0.95
            and candidate["mean_long_precision"] >= 0.98
        )
        report.append(candidate)
        print(json.dumps({"candidate": index + 1, "of": 32, **candidate}), flush=True)
    eligible = [candidate for candidate in report if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("no candidate satisfied cross-validation precision constraints")
    selected = sorted(
        eligible,
        key=lambda candidate: (
            -candidate["mean_utility"],
            candidate["utility_variance"],
            candidate["mean_artifact_bytes"],
        ),
    )[0]
    args.output.write_text(
        json.dumps({"schema": 1, "selected": selected, "candidates": report}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
