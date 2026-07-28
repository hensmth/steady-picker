#!/usr/bin/env python3
"""Build five deterministic cluster-grouped folds inside the training split."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def records(rows: list[dict]) -> str:
    return "".join(f"__label__{row['final_label']} {row['prompt']}\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds != 5:
        raise ValueError("settings-v2 requires exactly five folds")
    labels = {row["id"]: row for row in map(json.loads, args.labelled.read_text().splitlines())}
    membership = list(map(json.loads, args.membership.read_text().splitlines()))
    training_ids = {row["id"] for row in membership if row["split"] == "train"}
    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for row_id in training_ids:
        row = labels[row_id]
        if row["unresolved"]:
            raise RuntimeError("unresolved row entered training")
        clusters[row["cluster_id"]].append(row)
    if sum(map(len, clusters.values())) != 3000:
        raise RuntimeError("training split must contain exactly 3,000 rows")

    cluster_items = list(clusters.items())
    random.Random(args.seed).shuffle(cluster_items)
    fold_clusters: list[list[tuple[str, list[dict]]]] = [[] for _ in range(args.folds)]
    fold_strata = [collections.Counter() for _ in range(args.folds)]
    fold_sizes = [0] * args.folds
    for item in sorted(cluster_items, key=lambda value: -len(value[1])):
        stratum = collections.Counter((row["source"], row["final_label"]) for row in item[1])
        destination = min(
            range(args.folds),
            key=lambda fold: (
                sum(fold_strata[fold][key] for key in stratum),
                fold_sizes[fold],
                fold,
            ),
        )
        fold_clusters[destination].append(item)
        fold_strata[destination].update(stratum)
        fold_sizes[destination] += len(item[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        validation = [row for _, rows in fold_clusters[fold] for row in rows]
        remainder = [
            item
            for other in range(args.folds)
            if other != fold
            for item in fold_clusters[other]
        ]
        random.Random(args.seed + fold).shuffle(remainder)
        buckets = {"fit": [], "probability": [], "conformal": [], "development": []}
        targets = {"fit": 1800, "probability": 200, "conformal": 200, "development": 200}
        for _, rows in sorted(remainder, key=lambda value: -len(value[1])):
            options = [
                name for name, target in targets.items()
                if len(buckets[name]) + len(rows) <= target
            ]
            if not options:
                options = [max(targets, key=lambda name: targets[name] - len(buckets[name]))]
            destination = max(options, key=lambda name: targets[name] - len(buckets[name]))
            buckets[destination].extend(rows)
        fold_dir = args.output_dir / f"fold-{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in {**buckets, "validation": validation}.items():
            (fold_dir / f"{name}.txt").write_text(records(rows), encoding="utf-8")
        print(json.dumps({"fold": fold + 1, **{name: len(rows) for name, rows in buckets.items()}, "validation": len(validation)}))


if __name__ == "__main__":
    main()
