#!/usr/bin/env python3
"""Create immutable cluster-grouped, source/label-stratified v2 splits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

TARGETS = {
    "train": 3000,
    "probability_calibration": 300,
    "conformal_calibration": 300,
    "policy_development": 400,
    "locked_test": 1000,
}
FIT_SPLITS = {"train", "probability_calibration", "conformal_calibration"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 5000:
        raise RuntimeError("expected exactly 5,000 labelled rows")
    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row)
    rng = random.Random(args.seed)
    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)
    cluster_items.sort(
        key=lambda item: (
            not any(row["unresolved"] for row in item[1]),
            -len(item[1]),
            item[0],
        )
    )
    assigned: dict[str, list[dict]] = {name: [] for name in TARGETS}
    strata: dict[str, collections.Counter] = {
        name: collections.Counter() for name in TARGETS
    }
    total_strata = collections.Counter(
        (row["source"], row["final_label"] or "unresolved") for row in rows
    )
    for _, cluster_rows in cluster_items:
        unresolved = any(row["unresolved"] for row in cluster_rows)
        options = [
            name
            for name, target in TARGETS.items()
            if len(assigned[name]) + len(cluster_rows) <= target
            and not (unresolved and name in FIT_SPLITS)
        ]
        if not options:
            raise RuntimeError("cluster grouping cannot satisfy exact split sizes")
        cluster_strata = collections.Counter(
            (row["source"], row["final_label"] or "unresolved") for row in cluster_rows
        )

        def score(name: str) -> tuple[float, int, str]:
            size_deficit = (TARGETS[name] - len(assigned[name])) / TARGETS[name]
            stratification = 0.0
            for stratum, count in cluster_strata.items():
                desired = total_strata[stratum] * TARGETS[name] / len(rows)
                stratification += max(0.0, desired - strata[name][stratum]) * count
            return (stratification + size_deficit, TARGETS[name] - len(assigned[name]), name)

        destination = max(options, key=score)
        assigned[destination].extend(cluster_rows)
        strata[destination].update(cluster_strata)
    sizes = {name: len(values) for name, values in assigned.items()}
    if sizes != TARGETS:
        raise RuntimeError(f"split sizes are not exact: {sizes}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership: list[dict] = []
    hashes: dict[str, str] = {}
    for name, split_rows in assigned.items():
        split_rows.sort(key=lambda row: row["id"])
        labelled = "".join(
            f"__label__{row['final_label']} {row['prompt']}\n"
            for row in split_rows
            if not row["unresolved"]
        )
        path = args.output_dir / f"{name}.txt"
        path.write_text(labelled, encoding="utf-8")
        hashes[name] = hashlib.sha256(labelled.encode()).hexdigest()
        membership.extend(
            {
                "id": row["id"],
                "cluster_id": row["cluster_id"],
                "source": row["source"],
                "final_label": row["final_label"],
                "unresolved": row["unresolved"],
                "split": name,
            }
            for row in split_rows
        )
    membership.sort(key=lambda row: row["id"])
    membership_body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in membership)
    (args.output_dir / "membership.jsonl").write_text(membership_body, encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "targets": TARGETS,
        "file_sha256": hashes,
        "membership_sha256": hashlib.sha256(membership_body.encode()).hexdigest(),
        "grouping": "normalized token 3-gram Jaccard clusters",
        "stratification": ["source", "final_label"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
