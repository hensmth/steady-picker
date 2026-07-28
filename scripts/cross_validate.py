#!/usr/bin/env python3
"""Run the fixed settings-v2 hyperparameter grid and select one candidate."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
        "short_accepted": accepted["short"][0],
        "short_correct": accepted["short"][1],
        "long_accepted": accepted["long"][0],
        "long_correct": accepted["long"][1],
    }


def persist(path: Path, candidates: list[dict], selected: dict | None = None) -> None:
    report = {
        "schema": 1,
        "gate_passed": selected is not None,
        "selected": selected,
        "candidates": candidates,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        raise RuntimeError("--workers must be between 1 and 5")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    candidates = itertools.product(
        (100_000, 200_000),
        (1, 8),
        ((3, 5), (3, 6)),
        (0.05, 0.1),
        (1.0, 2.0),
    )
    report = []
    for index, (bucket, dimension, ngrams, learning_rate, positive_weight_scale) in enumerate(candidates):
        def run_fold(fold: int) -> tuple[dict, int]:
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
                "--epochs", "25", "--l2", "0.0001",
                "--positive-weight-scale", str(positive_weight_scale),
                "--source-manifest-sha256", args.source_manifest_sha256,
                "--training-code-commit", args.training_code_commit,
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            metric = evaluate(args.binary, artifact, rows(directory / "validation.txt"))
            size = artifact.stat().st_size
            artifact.unlink()
            return metric, size

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            fold_results = list(executor.map(run_fold, range(1, 6)))
        fold_metrics = [metric for metric, _ in fold_results]
        sizes = [size for _, size in fold_results]
        utilities = [metric["utility"] for metric in fold_metrics]
        short_accepted = sum(metric["short_accepted"] for metric in fold_metrics)
        short_correct = sum(metric["short_correct"] for metric in fold_metrics)
        long_accepted = sum(metric["long_accepted"] for metric in fold_metrics)
        long_correct = sum(metric["long_correct"] for metric in fold_metrics)
        candidate = {
            "bucket": bucket, "dimension": dimension, "min_ngram": ngrams[0],
            "max_ngram": ngrams[1], "learning_rate": learning_rate, "epochs": 25,
            "l2": 0.0001, "positive_weight_scale": positive_weight_scale,
            "mean_utility": statistics.mean(utilities),
            "utility_variance": statistics.pvariance(utilities),
            "mean_short_precision": statistics.mean(metric["short_precision"] for metric in fold_metrics),
            "mean_long_precision": statistics.mean(metric["long_precision"] for metric in fold_metrics),
            "mean_coverage": statistics.mean(metric["coverage"] for metric in fold_metrics),
            "mean_artifact_bytes": statistics.mean(sizes),
            "pooled_short_accepted": short_accepted,
            "pooled_short_precision": short_correct / short_accepted if short_accepted else 0,
            "pooled_long_accepted": long_accepted,
            "pooled_long_precision": long_correct / long_accepted if long_accepted else 0,
            "folds": fold_metrics,
        }
        candidate["eligible"] = (
            candidate["pooled_short_precision"] >= 0.95
            and candidate["pooled_long_precision"] >= 0.98
            and candidate["pooled_short_accepted"] >= 60
            and candidate["pooled_long_accepted"] >= 60
            and min(metric["short_accepted"] for metric in fold_metrics) >= 10
            and min(metric["long_accepted"] for metric in fold_metrics) >= 10
        )
        report.append(candidate)
        persist(args.output, report)
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
    persist(args.output, report, selected)


if __name__ == "__main__":
    main()
