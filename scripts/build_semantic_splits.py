#!/usr/bin/env python3
"""Create deterministic cluster-grouped 14k semantic-training splits."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


TARGETS = {
    "train": 10_000,
    "probability_calibration": 1_000,
    "conformal_calibration": 1_000,
    "policy_development": 2_000,
}


def label_key(row: dict) -> str:
    return row["final_label"] if not row["unresolved"] else "ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 14_000:
        raise RuntimeError(f"expected 14,000 labelled rows, got {len(rows)}")
    if [int(row["id"]) for row in rows] != list(range(14_000)):
        raise RuntimeError("development labels are not in canonical ID order")
    for row in rows:
        votes = row.get("structured_votes")
        if not isinstance(votes, list) or len(votes) != 3:
            raise RuntimeError("every row must contain exactly three structured votes")
        if row["unresolved"]:
            if row.get("final_label") is not None:
                raise RuntimeError("ambiguous row contains a final label")
        elif row.get("final_label") not in {"short", "medium", "long"}:
            raise RuntimeError("resolved row has an invalid final label")

    counts = collections.Counter(
        row["final_label"] for row in rows if not row["unresolved"]
    )
    if counts["short"] < 300 or counts["long"] < 300:
        raise RuntimeError("unanimous short/long development support is insufficient")

    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row)
    total_strata = collections.Counter(
        (
            row["source"],
            label_key(row),
            row.get("curriculum_stratum", row.get("corpus_partition", "existing")),
        )
        for row in rows
    )
    rng = random.Random(args.seed)
    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)
    cluster_items.sort(key=lambda item: (-len(item[1]), item[0]))

    assigned: dict[str, list[dict]] = {name: [] for name in TARGETS}
    strata: dict[str, collections.Counter] = {
        name: collections.Counter() for name in TARGETS
    }
    for _, cluster_rows in cluster_items:
        options = [
            name
            for name, target in TARGETS.items()
            if len(assigned[name]) + len(cluster_rows) <= target
        ]
        if not options:
            raise RuntimeError("cluster grouping cannot satisfy exact split sizes")
        cluster_strata = collections.Counter(
            (
                row["source"],
                label_key(row),
                row.get(
                    "curriculum_stratum",
                    row.get("corpus_partition", "existing"),
                ),
            )
            for row in cluster_rows
        )

        def score(name: str) -> tuple[float, int, str]:
            fit = 0.0
            for stratum, count in cluster_strata.items():
                desired = total_strata[stratum] * TARGETS[name] / len(rows)
                fit += max(0.0, desired - strata[name][stratum]) * count
            fill = (TARGETS[name] - len(assigned[name])) / TARGETS[name]
            return fit + fill, TARGETS[name] - len(assigned[name]), name

        destination = max(options, key=score)
        assigned[destination].extend(cluster_rows)
        strata[destination].update(cluster_strata)

    sizes = {name: len(values) for name, values in assigned.items()}
    if sizes != TARGETS:
        raise RuntimeError(f"semantic split sizes are not exact: {sizes}")
    for name, split_rows in assigned.items():
        split_counts = collections.Counter(
            row["final_label"] for row in split_rows if not row["unresolved"]
        )
        minimum = 150 if name == "train" else 30
        if split_counts["short"] < minimum or split_counts["long"] < minimum:
            raise RuntimeError(f"{name} has insufficient unanimous override support")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership: list[dict] = []
    hashes: dict[str, str] = {}
    for name, split_rows in assigned.items():
        split_rows.sort(key=lambda row: int(row["id"]))
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows)
        (args.output_dir / f"{name}.jsonl").write_text(body, encoding="utf-8")
        hashes[name] = hashlib.sha256(body.encode()).hexdigest()
        membership.extend(
            {
                "id": int(row["id"]),
                "cluster_id": row["cluster_id"],
                "source": row["source"],
                "label": label_key(row),
                "split": name,
            }
            for row in split_rows
        )
    membership.sort(key=lambda row: row["id"])
    membership_body = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in membership
    )
    (args.output_dir / "membership.jsonl").write_text(
        membership_body, encoding="utf-8"
    )
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "targets": TARGETS,
        "grouping": "normalized token 3-gram Jaccard clusters",
        "stratification": [
            "source",
            "unanimous_label_or_ambiguous",
            "curriculum_stratum",
        ],
        "sha256": {
            **hashes,
            "membership": hashlib.sha256(membership_body.encode()).hexdigest(),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
