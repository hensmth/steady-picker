#!/usr/bin/env python3
"""Evaluate locked/FETV/audit labels and enforce the immutable release gates."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

LABEL_DURATION = {"short": 2, "medium": 4, "long": 6}


def labelled_rows(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.startswith("{"):
            value = json.loads(line)
            if int(value["id"]) != index:
                raise RuntimeError("labelled JSONL changed canonical row order")
            rows.append(
                {
                    "id": index,
                    "label": (
                        "unresolved"
                        if value.get("unresolved", False)
                        else value["final_label"]
                    ),
                    "prompt": value["prompt"],
                }
            )
        else:
            label, prompt = line.split(" ", 1)
            rows.append(
                {
                    "id": index,
                    "label": label.removeprefix("__label__"),
                    "prompt": prompt,
                }
            )
    return rows


def predict(binary: Path, model: Path, rows: list[dict]) -> list[dict]:
    requests = "".join(
        json.dumps({"prompt": row["prompt"], "mode": "text-to-video"}) + "\n"
        for row in rows
    )
    result = subprocess.run(
        [str(binary), "predict", "--model", str(model), "--profile", "quota-safe-v2"],
        input=requests,
        text=True,
        capture_output=True,
        check=True,
    )
    predictions = [json.loads(line) for line in result.stdout.splitlines()]
    if len(predictions) != len(rows):
        raise RuntimeError("predictor changed row count")
    for prediction in predictions:
        if (
            prediction.get("model_version") != "settings-v2"
            or prediction.get("profile_version") != "2"
            or prediction.get("policy_version") != "quota-safe-v2"
            or len(prediction.get("artifact_sha256", "")) != 64
        ):
            raise RuntimeError("predictor returned invalid v5 provenance")
    return predictions


def predict_v1(binary: Path, model: Path, rows: list[dict]) -> list[dict]:
    requests = "".join(
        json.dumps({"prompt": row["prompt"], "mode": "text-to-video"}) + "\n"
        for row in rows
    )
    result = subprocess.run(
        [str(binary), "predict", "--model", str(model), "--model-version", "v1"],
        input=requests,
        text=True,
        capture_output=True,
        check=True,
    )
    predictions = [json.loads(line) for line in result.stdout.splitlines()]
    if len(predictions) != len(rows):
        raise RuntimeError("v1 predictor changed row count")
    for prediction in predictions:
        if prediction.get("model_version") != "v1":
            raise RuntimeError("v1 predictor returned invalid provenance")
        prediction["duration_source"] = prediction["source"]
        prediction["estimated_cost_microusd"] = prediction["duration"] * 50_000
    return predictions


def binomial_tail(successes: int, trials: int, probability: float) -> float:
    if successes == 0:
        return 1.0
    terms = []
    for value in range(successes, trials + 1):
        log_probability = (
            math.lgamma(trials + 1)
            - math.lgamma(value + 1)
            - math.lgamma(trials - value + 1)
            + value * math.log(max(probability, 1e-300))
            + (trials - value) * math.log(max(1 - probability, 1e-300))
        )
        terms.append(log_probability)
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in terms)


def exact_lower_bound(successes: int, trials: int, alpha: float = 0.05) -> float:
    if successes == 0 or trials == 0:
        return 0.0
    low, high = 0.0, successes / trials
    for _ in range(80):
        midpoint = (low + high) / 2
        if binomial_tail(successes, trials, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def selective_utility(label: str, prediction: dict) -> float:
    """Quota-first utility: correct accepted override +1, harmful short -4, wasteful long -2."""
    source = prediction["duration_source"]
    if source != "model":
        return 0.0
    if label == "unresolved":
        return -4.0 if prediction["duration"] == 2 else -2.0
    expected = LABEL_DURATION[label]
    actual = prediction["duration"]
    if actual == expected:
        return 1.0
    if actual < expected:
        return -4.0
    return -2.0


def metrics(rows: list[dict], predictions: list[dict]) -> dict:
    accepted = {"short": [0, 0], "long": [0, 0]}
    costs = []
    utility = 0.0
    risk_curve = []
    candidates = []
    for row, prediction in zip(rows, predictions):
        costs.append(prediction["estimated_cost_microusd"])
        utility += selective_utility(row["label"], prediction)
        if prediction["duration_source"] == "model":
            predicted = "short" if prediction["duration"] == 2 else "long"
            if predicted in accepted:
                accepted[predicted][0] += 1
                accepted[predicted][1] += row["label"] == predicted
                candidates.append((prediction["confidence"], row["label"] == predicted))
    for threshold in [value / 100 for value in range(80, 100)]:
        subset = [correct for confidence, correct in candidates if confidence >= threshold]
        risk_curve.append(
            {
                "threshold": threshold,
                "coverage": len(subset) / len(rows),
                "risk": 1 - sum(subset) / len(subset) if subset else 0,
            }
        )
    per_class = {}
    for label, (count, correct) in accepted.items():
        per_class[label] = {
            "accepted": count,
            "correct": correct,
            "precision": correct / count if count else 0,
            "exact_one_sided_95_lower": exact_lower_bound(correct, count),
        }
    return {
        "examples": len(rows),
        "override_coverage": sum(value[0] for value in accepted.values()) / len(rows),
        "per_class": per_class,
        "mean_projected_spend_microusd": sum(costs) / len(costs),
        "mean_selective_utility": utility / len(rows),
        "risk_coverage_curve": risk_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--fetv-labelled", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--v1-binary", type=Path, required=True)
    parser.add_argument("--v1-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-marker", type=Path, required=True)
    args = parser.parse_args()

    locked_rows = labelled_rows(args.locked_test)
    if len(locked_rows) != 1000:
        raise RuntimeError("locked test must contain exactly 1,000 rows")
    locked = metrics(locked_rows, predict(args.binary, args.model, locked_rows))
    v1 = metrics(
        locked_rows,
        predict_v1(args.v1_binary, args.v1_model, locked_rows),
    )
    fetv_rows = labelled_rows(args.fetv_labelled)
    if len(fetv_rows) != 619:
        raise RuntimeError("FETV evaluation must contain exactly 619 rows")
    fetv = metrics(fetv_rows, predict(args.binary, args.model, fetv_rows))
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    deterministic_utility = 0.0
    gates = {
        "short_point_precision": locked["per_class"]["short"]["precision"] >= 0.95,
        "short_exact_lower": locked["per_class"]["short"]["exact_one_sided_95_lower"] >= 0.90,
        "short_accepted": locked["per_class"]["short"]["accepted"] >= 60,
        "long_point_precision": locked["per_class"]["long"]["precision"] >= 0.98,
        "long_exact_lower": locked["per_class"]["long"]["exact_one_sided_95_lower"] >= 0.95,
        "long_accepted": locked["per_class"]["long"]["accepted"] >= 60,
        "coverage": locked["override_coverage"] >= 0.12,
        "spend": locked["mean_projected_spend_microusd"] <= 200_000,
        "beats_deterministic": locked["mean_selective_utility"] > deterministic_utility,
        "beats_v1": locked["mean_selective_utility"] > v1["mean_selective_utility"],
        "audit_agreement": audit["agreement"] >= 0.90,
        "audit_kappa": audit["weighted_kappa"] >= 0.80,
        "audit_no_severe_short": audit["severe_short_under_duration_errors"] == 0,
        "audit_no_unnecessary_long": audit["unnecessary_long_overrides"] == 0,
    }
    report = {
        "schema": 1,
        "locked_test": locked,
        "external_fetv": fetv,
        "independent_audit": audit,
        "settings_v1": v1,
        "deterministic_only_mean_selective_utility": deterministic_utility,
        "gates": gates,
        "passed": all(gates.values()),
        "conformal_scope_note": (
            "Coverage is reported in-domain and empirically on FETV; no guarantee "
            "is claimed under distribution shift."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.gate_marker.unlink(missing_ok=True)
    if not report["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"release gates failed: {', '.join(failed)}")
    args.gate_marker.write_text("all locked gates passed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
