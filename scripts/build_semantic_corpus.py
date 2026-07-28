#!/usr/bin/env python3
"""Build the 14k development corpus for semantic settings-v2 retraining."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from build_corpus import (
    ACTION_CUE,
    DIFFUSIONDB_REVISION,
    TEMPORAL_CUE,
    VIDEOUFO_REVISION,
    excluded_near_duplicate,
    fetch_diffusiondb,
    fetch_videoufo,
    file_sha256,
    load_exclusions,
    near_duplicate_clusters,
    token_trigrams,
)


FRESH_QUOTAS = {
    ("high", "VideoUFO"): 500,
    ("high", "DiffusionDB"): 3500,
    ("hard_negative_candidate", "VideoUFO"): 1900,
    ("hard_negative_candidate", "DiffusionDB"): 100,
    ("broad", "VideoUFO"): 2600,
    ("broad", "DiffusionDB"): 1400,
}


def fresh_stratum(row: dict) -> str | None:
    score = int(row["temporal_complexity"])
    if score >= 2:
        return "high"
    if score == 1 and (
        TEMPORAL_CUE.search(row["prompt"]) or ACTION_CUE.search(row["prompt"])
    ):
        return "hard_negative_candidate"
    if score == 0:
        return "broad"
    return None


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-labelled", type=Path, required=True)
    parser.add_argument("--existing-membership", type=Path, required=True)
    parser.add_argument("--sealed-holdout-prompts", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--fresh-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--videoufo-pool", type=int, default=120_000)
    parser.add_argument("--videoufo-cache-only", action="store_true")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()

    for dataset, expected in (
        ("WenhaoWang/VideoUFO", VIDEOUFO_REVISION),
        ("poloclub/diffusiondb", DIFFUSIONDB_REVISION),
    ):
        with urllib.request.urlopen(
            f"https://huggingface.co/api/datasets/{dataset}", timeout=30
        ) as response:
            revision = json.load(response)["sha"]
        if revision != expected:
            raise RuntimeError(
                f"{dataset} revision changed; review and pin before rebuilding"
            )

    labelled = {
        int(row["id"]): row for row in read_rows(args.existing_labelled)
    }
    membership = read_rows(args.existing_membership)
    existing = [
        labelled[int(member["id"])]
        for member in membership
        if member["split"] != "locked_test"
    ]
    if len(existing) != 4000:
        raise RuntimeError(f"expected 4,000 existing non-locked rows, got {len(existing)}")

    holdout = read_rows(args.sealed_holdout_prompts)
    if len(holdout) != 1000:
        raise RuntimeError("sealed holdout must contain exactly 1,000 prompts")

    exclusions = [*args.exclude, args.existing_labelled, args.sealed_holdout_prompts]
    excluded_exact, excluded_index, exclusion_manifests = load_exclusions(exclusions)

    video = fetch_videoufo(
        args.videoufo_pool,
        args.cache / "videoufo-pages",
        args.seed,
        args.videoufo_cache_only,
    )
    diffusion = fetch_diffusiondb(args.cache, 120_000, args.seed + 1)
    candidates = video + diffusion
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in candidates:
        stratum = fresh_stratum(row)
        if stratum is None or excluded_near_duplicate(
            row["prompt"], excluded_exact, excluded_index
        ):
            continue
        groups[(stratum, row["source"])].append(row)

    rng = random.Random(args.seed)
    for values in groups.values():
        values.sort(key=lambda row: (row["source_id"], row["prompt"]))
        rng.shuffle(values)

    selected: list[dict] = []
    selected_exact = set(excluded_exact)
    selected_index = collections.defaultdict(list)
    for gram, values in excluded_index.items():
        selected_index[gram].extend(values)

    for key, quota in FRESH_QUOTAS.items():
        accepted = 0
        for row in groups[key]:
            if excluded_near_duplicate(
                row["prompt"], selected_exact, selected_index
            ):
                continue
            row = {**row, "curriculum_stratum": key[0]}
            selected.append(row)
            accepted += 1
            selected_exact.add(row["prompt"].casefold())
            grams = token_trigrams(row["prompt"])
            for gram in grams:
                selected_index[gram].append(grams)
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError(f"{key} yielded {accepted} rows; need {quota}")

    if len(selected) != 10_000:
        raise RuntimeError(f"fresh selection produced {len(selected)} rows")
    rng.shuffle(selected)
    near_duplicate_clusters(selected)
    for index, row in enumerate(selected):
        row["id"] = index
    fresh_body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)

    development: list[dict] = []
    for row in existing:
        development.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "teacher_votes",
                    "structured_votes",
                    "structured_consensus",
                    "final_label",
                    "unresolved",
                }
            }
            | {"corpus_partition": "existing"}
        )
    development.extend(
        {**row, "corpus_partition": "fresh"} for row in selected
    )
    near_duplicate_clusters(development)
    for index, row in enumerate(development):
        row["id"] = index
    development_body = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in development
    )

    args.development_output.parent.mkdir(parents=True, exist_ok=True)
    args.fresh_output.write_text(fresh_body, encoding="utf-8")
    args.development_output.write_text(development_body, encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "rows": {
            "existing_non_locked": 4000,
            "fresh": 10_000,
            "sealed_holdout": 1000,
            "total": 15_000,
        },
        "fresh_quotas": {
            f"{stratum}:{source}": count
            for (stratum, source), count in FRESH_QUOTAS.items()
        },
        "sources": {
            "VideoUFO": {
                "revision": VIDEOUFO_REVISION,
                "license": "CC-BY-4.0",
            },
            "DiffusionDB": {
                "revision": DIFFUSIONDB_REVISION,
                "license": "CC0-1.0",
                "raw_parquet_sha256": file_sha256(
                    args.cache / "diffusiondb.parquet"
                ),
            },
        },
        "sha256": {
            "fresh": hashlib.sha256(fresh_body.encode()).hexdigest(),
            "development": hashlib.sha256(development_body.encode()).hexdigest(),
            "sealed_holdout_prompts": hashlib.sha256(
                args.sealed_holdout_prompts.read_bytes()
            ).hexdigest(),
        },
        "prior_corpus_exclusions": exclusion_manifests,
        "near_duplicate_policy": "normalized token 3-gram Jaccard >= 0.85",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
