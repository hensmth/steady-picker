#!/usr/bin/env python3
"""Select a sealed, support-enriched holdout without using teacher labels."""

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
    DEPENDENT_ACTION,
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


QUOTAS = {
    ("short_candidate", "VideoUFO"): 225,
    ("short_candidate", "DiffusionDB"): 175,
    ("bridge", "VideoUFO"): 50,
    ("bridge", "DiffusionDB"): 50,
    ("long_candidate", "VideoUFO"): 200,
    ("long_candidate", "DiffusionDB"): 300,
}


def action_count(prompt: str) -> int:
    return len(ACTION_CUE.findall(prompt))


def stratum(row: dict) -> str | None:
    prompt = row["prompt"]
    score = int(row["temporal_complexity"])
    actions = action_count(prompt)
    temporal = bool(TEMPORAL_CUE.search(prompt))
    dependent = bool(DEPENDENT_ACTION.search(prompt))
    words = len(prompt.split())
    if score <= 1 and not temporal and not dependent and actions <= 1 and words <= 80:
        return "short_candidate"
    if score >= 3 and temporal:
        return "long_candidate"
    if 2 <= score <= 3:
        return "bridge"
    return None


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
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

    excluded_exact, excluded_index, exclusion_manifests = load_exclusions(
        args.exclude
    )
    candidates = fetch_videoufo(
        args.videoufo_pool,
        args.cache / "videoufo-pages",
        args.seed,
        args.videoufo_cache_only,
    )
    candidates.extend(fetch_diffusiondb(args.cache, 120_000, args.seed + 1))

    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in candidates:
        candidate_stratum = stratum(row)
        if candidate_stratum is None or excluded_near_duplicate(
            row["prompt"], excluded_exact, excluded_index
        ):
            continue
        groups[(candidate_stratum, row["source"])].append(row)

    rng = random.Random(args.seed)
    for key, values in groups.items():
        values.sort(key=lambda row: (row["source_id"], row["prompt"]))
        rng.shuffle(values)
        if key[0] == "long_candidate":
            values.sort(key=lambda row: int(row["temporal_complexity"]), reverse=True)

    selected: list[dict] = []
    selected_exact = set(excluded_exact)
    selected_index: dict[
        tuple[str, str, str], list[frozenset[tuple[str, str, str]]]
    ] = collections.defaultdict(list)
    for gram, values in excluded_index.items():
        selected_index[gram].extend(values)

    for key, quota in QUOTAS.items():
        accepted = 0
        for row in groups[key]:
            if excluded_near_duplicate(
                row["prompt"], selected_exact, selected_index
            ):
                continue
            selected.append({**row, "holdout_stratum": key[0]})
            accepted += 1
            selected_exact.add(row["prompt"].casefold())
            grams = token_trigrams(row["prompt"])
            for gram in grams:
                selected_index[gram].append(grams)
            if accepted == quota:
                break
        if accepted != quota:
            raise RuntimeError(f"{key} yielded {accepted} rows; need {quota}")

    if len(selected) != 1_000:
        raise RuntimeError(f"holdout selection produced {len(selected)} rows")
    rng.shuffle(selected)
    near_duplicate_clusters(selected)
    for index, row in enumerate(selected):
        row["id"] = index
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "rows": 1_000,
        "selection_policy": (
            "pre-label deterministic temporal-cue strata; teacher labels unused"
        ),
        "quotas": {
            f"{candidate_stratum}:{source}": count
            for (candidate_stratum, source), count in QUOTAS.items()
        },
        "criteria": {
            "short_candidate": (
                "score <= 1, at most one action cue, no progression or dependent "
                "action cue, at most 80 whitespace-delimited words"
            ),
            "bridge": "temporal complexity score 2 through 3",
            "long_candidate": (
                "score >= 3 with an explicit progression or transformation cue"
            ),
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
        "exclusions": exclusion_manifests,
        "near_duplicate_policy": "normalized token 3-gram Jaccard >= 0.85",
        "output_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": 1_000,
                "sha256": manifest["output_sha256"],
                "strata": manifest["quotas"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
