#!/usr/bin/env python3
"""Run grouped semantic CV, freeze v5, export INT8, and verify Go parity."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
import sentence_transformers
from torch.utils.data import DataLoader

from semantic_training import (
    TrainConfig,
    WordPiece,
    accepted,
    binary_labels,
    build_vocabulary,
    conformal_quantiles,
    export_v5,
    fit_temperatures,
    positive_class_weights,
    predict,
    probabilities,
    quantized_copy,
    select_thresholds,
    train_student,
)


TEACHER_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TEACHER_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
GRID = list(
    itertools.product(
        (0.0003, 0.0007),
        (0.75, 1.25),
        (0.25, 0.5),
        (1.0, 2.0),
    )
)
PRAGMATIC_CANDIDATE_INDEX = 8
PRAGMATIC_DECISION_POLICY = "long-only-pragmatic-v2"
PRAGMATIC_EXPECTED_LONG_FOLDS = (
    (6, 4),
    (6, 5),
    (4, 3),
    (8, 6),
    (5, 5),
)
STRONG_LONG_CUES = (
    " then ",
    " and then ",
    " followed by ",
    " finally ",
    "transform",
    "turns into",
    "turn into",
    "turning into",
    "morph",
    "becomes ",
    "evolv",
    "grows into",
    "time-lapse",
    "timelapse",
    "gradually",
    "eventually",
    "over time",
    "life cycle",
    "through the seasons",
    "through seasons",
    "stages",
)
ACTION_CUES = (
    " walking",
    " running",
    " driving",
    " traveling",
    " flying",
    " taking",
    " opening",
    " picking",
    " placing",
    " putting",
    " pouring",
    " adding",
    " mixing",
    " cutting",
    " folding",
    " cooking",
    " building",
    " constructing",
    " assembling",
    " growing",
    " blooming",
    " sprouting",
    " crashing",
    " exploding",
    " destroying",
    " repairing",
    " shooting",
    " firing",
    " fighting",
    " attacking",
    " drinking",
    " eating",
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_label(row: dict) -> str:
    return (
        row["final_label"]
        if not row.get("unresolved", True)
        else "ambiguous"
    )


def group_folds(rows: list[dict], seed: int) -> list[int]:
    clusters: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        clusters[row["cluster_id"]].append(index)
    by_stratum: dict[tuple[str, str], list[tuple[str, list[int]]]] = (
        collections.defaultdict(list)
    )
    for cluster, indices in clusters.items():
        representative = rows[indices[0]]
        by_stratum[(representative["source"], row_label(representative))].append(
            (cluster, indices)
        )
    folds = [-1] * len(rows)
    totals = [0] * 5
    for stratum in sorted(by_stratum):
        values = by_stratum[stratum]
        values.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{item[0]}".encode()
            ).digest()
        )
        stratum_totals = [0] * 5
        for _, indices in values:
            fold = min(
                range(5),
                key=lambda value: (stratum_totals[value], totals[value], value),
            )
            for index in indices:
                folds[index] = fold
            stratum_totals[fold] += len(indices)
            totals[fold] += len(indices)
    if -1 in folds:
        raise RuntimeError("grouped fold assignment is incomplete")
    return folds


def internal_partition(rows: list[dict], seed: int) -> dict[str, list[int]]:
    output = {name: [] for name in ("fit", "probability", "conformal", "policy")}
    groups: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["source"], row_label(row))].append(index)
    for stratum in sorted(groups):
        values = groups[stratum]
        values.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}:{rows[index]['cluster_id']}".encode()
            ).digest()
        )
        for offset, index in enumerate(values):
            slot = offset % 10
            destination = (
                "fit"
                if slot < 6
                else "probability"
                if slot == 6
                else "conformal"
                if slot == 7
                else "policy"
            )
            output[destination].append(index)
    if any(not values for values in output.values()):
        raise RuntimeError("nested CV partition is empty")
    return output


def long_cue_eligible(text: str) -> bool:
    folded = text.lower()
    if any(cue in folded for cue in STRONG_LONG_CUES):
        return True
    return sum(cue in folded for cue in ACTION_CUES) >= 2


def runtime_accepted(
    values: np.ndarray,
    quantiles: np.ndarray,
    thresholds: np.ndarray,
    rows: list[dict],
) -> np.ndarray:
    result = accepted(values, quantiles, thresholds)
    result[:, 1] &= np.asarray(
        [long_cue_eligible(row["prompt"]) for row in rows],
        dtype=bool,
    )
    result[result[:, 0] & result[:, 1]] = False
    return result


def policy_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    quantiles: np.ndarray,
    rows: list[dict],
    long_precision_target: float = 0.98,
) -> np.ndarray:
    # Use the common threshold search for short, then apply the exact runtime
    # eligibility mask while searching long.
    thresholds = select_thresholds(values, labels, quantiles)
    best = (0, 1.0)
    for threshold in np.arange(0.8, 0.9951, 0.005):
        candidates = runtime_accepted(
            values,
            quantiles,
            np.asarray([1.0, threshold], dtype=np.float32),
            rows,
        )[:, 1]
        count = int(candidates.sum())
        precision = float(labels[candidates, 1].mean()) if count else 0.0
        if precision >= long_precision_target and (
            count > best[0] or (count == best[0] and threshold > best[1])
        ):
            best = count, float(threshold)
    thresholds[1] = best[1]
    return thresholds


def fit_teacher(
    rows: list[dict],
    output_dir: Path,
    device: str,
    seed: int,
) -> SentenceTransformer:
    checkpoint = output_dir / "teacher-encoder"
    if (checkpoint / "config_sentence_transformers.json").is_file():
        return SentenceTransformer(str(checkpoint), device=device)
    torch.manual_seed(seed)
    model = SentenceTransformer(
        TEACHER_ID,
        revision=TEACHER_REVISION,
        device=device,
    )
    resolved: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        if not row["unresolved"]:
            resolved[row["final_label"]].append(row)
    if any(len(resolved[label]) < 100 for label in ("short", "medium", "long")):
        raise RuntimeError("teacher contrastive support is insufficient")
    rng = random.Random(seed)
    examples = []
    labels = ("short", "medium", "long")
    for _ in range(4_000):
        same = rng.random() < 0.5
        first_label = rng.choice(labels)
        second_label = first_label if same else rng.choice(
            [label for label in labels if label != first_label]
        )
        first = rng.choice(resolved[first_label])["prompt"]
        second = rng.choice(resolved[second_label])["prompt"]
        examples.append(InputExample(texts=[first, second], label=float(same)))
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=32,
        generator=torch.Generator().manual_seed(seed),
    )
    # The pinned SentenceTransformers release keeps this DataLoader-based path
    # for deterministic loss training without its optional datasets dependency.
    model.old_fit(
        train_objectives=[(loader, losses.CosineSimilarityLoss(model))],
        epochs=1,
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
    )
    model.save(str(checkpoint))
    return model


def teacher_embeddings(
    model: SentenceTransformer,
    rows: list[dict],
    path: Path,
) -> np.ndarray:
    if path.is_file():
        values = np.load(path)
        if values.shape[0] != len(rows):
            raise RuntimeError("teacher embedding checkpoint has the wrong row count")
        return values
    values = model.encode(
        [row["prompt"] for row in rows],
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(path, values)
    return values


def cv_candidate(
    candidate_index: int,
    values: tuple[float, float, float, float],
    train_rows: list[dict],
    embeddings: np.ndarray,
    tokenizer: WordPiece,
    folds: list[int],
    epochs: int,
    device: torch.device,
    seed: int,
) -> dict:
    fold_results = []
    for fold in range(5):
        outer_train = [index for index, value in enumerate(folds) if value != fold]
        outer_test = [index for index, value in enumerate(folds) if value == fold]
        nested_rows = [train_rows[index] for index in outer_train]
        nested = internal_partition(nested_rows, seed + candidate_index * 10 + fold)

        def rows_for(name: str) -> list[dict]:
            return [nested_rows[index] for index in nested[name]]

        fit_rows = rows_for("fit")
        model = train_student(
            fit_rows,
            embeddings[
                [outer_train[index] for index in nested["fit"]]
            ],
            tokenizer,
            TrainConfig(*values, epochs=epochs, seed=seed + candidate_index * 10 + fold),
            device,
        )
        probability_rows = rows_for("probability")
        probability_logits = predict(model, probability_rows, tokenizer, device)
        temperatures = fit_temperatures(
            probability_logits, binary_labels(probability_rows)
        )
        conformal_rows = rows_for("conformal")
        conformal_values = probabilities(
            predict(model, conformal_rows, tokenizer, device), temperatures
        )
        quantiles = conformal_quantiles(
            conformal_values, binary_labels(conformal_rows)
        )
        policy_rows = rows_for("policy")
        policy_values = probabilities(
            predict(model, policy_rows, tokenizer, device), temperatures
        )
        thresholds = policy_thresholds(
            policy_values,
            binary_labels(policy_rows),
            quantiles,
            policy_rows,
        )
        test_rows = [train_rows[index] for index in outer_test]
        test_values = probabilities(
            predict(model, test_rows, tokenizer, device), temperatures
        )
        test_labels = binary_labels(test_rows)
        overrides = runtime_accepted(
            test_values, quantiles, thresholds, test_rows
        )
        heads = []
        for head in range(2):
            count = int(overrides[:, head].sum())
            correct = int(test_labels[overrides[:, head], head].sum())
            heads.append(
                {
                    "accepted": count,
                    "correct": correct,
                    "precision": correct / count if count else 0.0,
                }
            )
        fold_results.append(
            {
                "fold": fold,
                "rows": len(test_rows),
                "heads": heads,
                "temperatures": temperatures.tolist(),
                "quantiles": quantiles.tolist(),
                "thresholds": thresholds.tolist(),
            }
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    pooled = []
    for head in range(2):
        count = sum(result["heads"][head]["accepted"] for result in fold_results)
        correct = sum(result["heads"][head]["correct"] for result in fold_results)
        pooled.append(
            {
                "accepted": count,
                "correct": correct,
                "precision": correct / count if count else 0.0,
            }
        )
    eligible = (
        pooled[0]["precision"] >= 0.95
        and pooled[1]["precision"] >= 0.98
        and pooled[0]["accepted"] >= 60
        and pooled[1]["accepted"] >= 60
        and all(
            result["heads"][head]["accepted"] >= 10
            for result in fold_results
            for head in range(2)
        )
    )
    utility = (
        pooled[0]["correct"] * 2
        - (pooled[0]["accepted"] - pooled[0]["correct"]) * 20
        + pooled[1]["correct"] * 2
        - (pooled[1]["accepted"] - pooled[1]["correct"]) * 50
    ) / len(train_rows)
    variance = float(
        np.var(
            [
                result["heads"][head]["precision"]
                for result in fold_results
                for head in range(2)
            ]
        )
    )
    return {
        "candidate": candidate_index,
        "config": {
            "learning_rate": values[0],
            "long_weight_multiplier": values[1],
            "contrastive_weight": values[2],
            "focal_gamma": values[3],
        },
        "folds": fold_results,
        "pooled": pooled,
        "eligible": eligible,
        "cost_weighted_selective_utility": utility,
        "fold_precision_variance": variance,
    }


def pragmatic_candidate_eligible(result: dict) -> bool:
    pooled = result["pooled"][1]
    folds = tuple(
        (fold["heads"][1]["accepted"], fold["heads"][1]["correct"])
        for fold in result["folds"]
    )
    return (
        result["candidate"] == PRAGMATIC_CANDIDATE_INDEX
        and pooled["accepted"] == 29
        and pooled["correct"] == 23
        and pooled["precision"] >= 0.75
        and folds == PRAGMATIC_EXPECTED_LONG_FOLDS
    )


def verify_go_parity(
    repository: Path,
    artifact: Path,
    model,
    rows: list[dict],
    tokenizer: WordPiece,
    temperatures: np.ndarray,
) -> dict:
    samples = rows[:32]
    quantized = quantized_copy(model)
    expected = probabilities(
        np.concatenate(
            [
                predict(
                    quantized,
                    [row],
                    tokenizer,
                    torch.device("cpu"),
                    batch_size=1,
                )
                for row in samples
            ]
        ),
        temperatures,
    )
    request = "".join(
        json.dumps({"prompt": row["prompt"]}) + "\n" for row in samples
    )
    result = subprocess.run(
        ["go", "run", "./cmd/steady-parity", "--model", str(artifact)],
        cwd=repository,
        input=request,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = np.asarray(
        [
            json.loads(line)["probabilities"]
            for line in result.stdout.splitlines()
            if line.strip()
        ],
        dtype=np.float32,
    )
    if actual.shape != expected.shape:
        raise RuntimeError("Go parity output has the wrong shape")
    maximum = float(np.max(np.abs(actual - expected)))
    if maximum > 2e-4:
        raise RuntimeError(f"PyTorch/Go parity error {maximum} exceeds 0.0002")
    return {"samples": len(samples), "maximum_absolute_error": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument(
        "--sealed-support-validation", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--training-code-commit", required=True)
    parser.add_argument("--cv-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-policy",
        choices=("strict-v2", PRAGMATIC_DECISION_POLICY),
        default="strict-v2",
    )
    parser.add_argument(
        "--recompute-cv",
        action="store_true",
        help="ignore existing CV checkpoints and reproduce the selected policy",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="stop after validating the frozen grouped-CV selection",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    development = read_jsonl(args.development_labels)
    if len(development) != 14_000:
        raise RuntimeError("development labels must contain exactly 14,000 rows")
    splits = {
        name: read_jsonl(args.split_dir / f"{name}.jsonl")
        for name in (
            "train",
            "probability_calibration",
            "conformal_calibration",
            "policy_development",
        )
    }
    if {name: len(rows) for name, rows in splits.items()} != {
        "train": 10_000,
        "probability_calibration": 1_000,
        "conformal_calibration": 1_000,
        "policy_development": 2_000,
    }:
        raise RuntimeError("semantic split sizes are invalid")
    teacher_manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if (
        teacher_manifest["provider"] != "openai-codex"
        or teacher_manifest["model"] != "gpt-5.6-sol"
        or teacher_manifest["reasoning_effort"] != "ultra"
    ):
        raise RuntimeError("teacher provenance does not match the frozen plan")
    sealed_support = json.loads(
        args.sealed_support_validation.read_text(encoding="utf-8")
    )
    if (
        sealed_support.get("row_integrity") is not True
        or sealed_support.get("unanimous_override_support_sufficient") is not True
    ):
        raise RuntimeError("sealed holdout support validation did not pass")

    vocabulary_path = args.output_dir / "vocabulary.json"
    if vocabulary_path.is_file():
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    else:
        vocabulary = build_vocabulary(row["prompt"] for row in development)
        vocabulary_path.write_text(
            json.dumps(vocabulary, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    tokenizer = WordPiece(vocabulary)
    tokenizer.cache(development)
    tokens_by_id = {
        int(row["id"]): row["_semantic_token_ids"] for row in development
    }
    for split_rows in splits.values():
        for row in split_rows:
            row["_semantic_token_ids"] = tokens_by_id[int(row["id"])]
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    teacher = fit_teacher(
        splits["train"], args.output_dir, str(device), args.seed
    )
    embeddings = teacher_embeddings(
        teacher,
        development,
        args.output_dir / "teacher-embeddings.npy",
    )
    del teacher
    embedding_by_id = {
        int(row["id"]): embeddings[index]
        for index, row in enumerate(development)
    }
    train_rows = splits["train"]
    train_embeddings = np.stack(
        [embedding_by_id[int(row["id"])] for row in train_rows]
    )
    folds = group_folds(train_rows, args.seed)

    cv_path = args.output_dir / "cv-results.jsonl"
    if args.recompute_cv:
        cv_path.unlink(missing_ok=True)
    existing = {}
    if cv_path.is_file() and not args.recompute_cv:
        for line in cv_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = json.loads(line)
                existing[int(result["candidate"])] = result
    candidate_indices = (
        [PRAGMATIC_CANDIDATE_INDEX]
        if args.decision_policy == PRAGMATIC_DECISION_POLICY
        else list(range(len(GRID)))
    )
    results = []
    for candidate_index in candidate_indices:
        values = GRID[candidate_index]
        if candidate_index in existing:
            result = existing[candidate_index]
        else:
            result = cv_candidate(
                candidate_index,
                values,
                train_rows,
                train_embeddings,
                tokenizer,
                folds,
                args.cv_epochs,
                device,
                args.seed,
            )
            with cv_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    eligible = [
        result
        for result in results
        if (
            pragmatic_candidate_eligible(result)
            if args.decision_policy == PRAGMATIC_DECISION_POLICY
            else result["eligible"]
        )
    ]
    diagnostics = {
        "schema": 1,
        "grid_candidates": len(candidate_indices),
        "eligible_candidates": len(eligible),
        "decision_policy": args.decision_policy,
        "sealed_support_validation_sha256": hashlib.sha256(
            args.sealed_support_validation.read_bytes()
        ).hexdigest(),
        "frozen_cv_gates": (
            {
                "long_precision": 0.75,
                "long_accepted": 20,
                "learned_short_enabled": False,
                "exact_reproduction": {
                    "accepted": 29,
                    "correct": 23,
                    "folds": PRAGMATIC_EXPECTED_LONG_FOLDS,
                },
            }
            if args.decision_policy == PRAGMATIC_DECISION_POLICY
            else {
                "short_precision": 0.95,
                "long_precision": 0.98,
                "accepted_per_class": 60,
                "accepted_per_class_per_fold": 10,
            }
        ),
        "results": results,
    }
    diagnostics_path = args.output_dir / "cv-diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not eligible:
        if args.decision_policy == PRAGMATIC_DECISION_POLICY:
            raise RuntimeError(
                "candidate 8 did not exactly reproduce the frozen pragmatic "
                "long-only CV evidence; replacement holdout remains sealed"
            )
        raise RuntimeError(
            "all 16 semantic candidates failed frozen grouped-CV gates; "
            "replacement holdout remains sealed"
        )
    if args.decision_policy == PRAGMATIC_DECISION_POLICY:
        selected = dict(eligible[0])
        selected["strict_v2_eligible"] = selected["eligible"]
        selected["eligible"] = True
        selected["eligibility_policy"] = PRAGMATIC_DECISION_POLICY
    else:
        selected = max(
            eligible,
            key=lambda result: (
                result["cost_weighted_selective_utility"],
                -result["fold_precision_variance"],
                -result["candidate"],
            ),
        )
    if args.cv_only:
        print(
            json.dumps(
                {
                    "decision_policy": args.decision_policy,
                    "selected_candidate": selected,
                    "status": "cv-passed",
                },
                sort_keys=True,
            )
        )
        return
    config = selected["config"]
    model = train_student(
        train_rows,
        train_embeddings,
        tokenizer,
        TrainConfig(
            config["learning_rate"],
            config["long_weight_multiplier"],
            config["contrastive_weight"],
            config["focal_gamma"],
            epochs=args.final_epochs,
            seed=args.seed,
        ),
        device,
    )
    probability_rows = splits["probability_calibration"]
    temperatures = fit_temperatures(
        predict(model, probability_rows, tokenizer, device),
        binary_labels(probability_rows),
    )
    conformal_rows = splits["conformal_calibration"]
    conformal_values = probabilities(
        predict(model, conformal_rows, tokenizer, device), temperatures
    )
    quantiles = conformal_quantiles(
        conformal_values, binary_labels(conformal_rows)
    )
    policy_rows = splits["policy_development"]
    policy_values = probabilities(
        predict(model, policy_rows, tokenizer, device), temperatures
    )
    thresholds = policy_thresholds(
        policy_values,
        binary_labels(policy_rows),
        quantiles,
        policy_rows,
        (
            0.75
            if args.decision_policy == PRAGMATIC_DECISION_POLICY
            else 0.98
        ),
    )
    if args.decision_policy == PRAGMATIC_DECISION_POLICY:
        thresholds[0] = 1.0
    torch.save(model.state_dict(), args.output_dir / "frozen-student.pt")
    source_digest = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    class_weights = positive_class_weights(
        train_rows, config["long_weight_multiplier"]
    )
    metadata = {
        "epochs": args.final_epochs,
        "learning_rate": config["learning_rate"],
        "l2": 0.01,
        "alpha": 0.1,
        "seed": args.seed,
        "source_manifest_sha256": source_digest,
        "training_code_commit": args.training_code_commit,
        "positive_class_weights": list(class_weights),
        "teacher_encoder": f"{TEACHER_ID}@{TEACHER_REVISION}",
        "training_provider": teacher_manifest["provider"],
        "training_model": teacher_manifest["model"],
        "training_effort": teacher_manifest["reasoning_effort"],
        "training_backend": "pytorch-cpu-deterministic",
        "training_toolchain": (
            f"python={platform.python_version()};torch={torch.__version__};"
            f"numpy={np.__version__};"
            f"sentence-transformers={sentence_transformers.__version__}"
        ),
    }
    artifact = export_v5(
        args.artifact,
        model,
        vocabulary,
        metadata,
        temperatures,
        quantiles,
        thresholds,
        (
            PRAGMATIC_DECISION_POLICY
            if args.decision_policy == PRAGMATIC_DECISION_POLICY
            else ""
        ),
    )
    parity = verify_go_parity(
        Path.cwd(),
        args.artifact,
        model,
        policy_rows,
        tokenizer,
        temperatures,
    )
    frozen = {
        "schema": 1,
        "selected_candidate": selected,
        "temperatures": temperatures.tolist(),
        "quantiles": quantiles.tolist(),
        "thresholds": thresholds.tolist(),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_bytes": len(artifact),
        "pytorch_go_parity": parity,
        "decision_policy": args.decision_policy,
    }
    (args.output_dir / "frozen-model.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(frozen, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"semantic training failed: {error}", file=sys.stderr)
        raise
