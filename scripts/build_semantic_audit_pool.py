#!/usr/bin/env python3
"""Build a fresh rights-cleared pool for the post-freeze 200-row audit."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from build_corpus import (
    DIFFUSIONDB_REVISION,
    VIDEOUFO_REVISION,
    excluded_near_duplicate,
    fetch_diffusiondb,
    fetch_videoufo,
    load_exclusions,
    near_duplicate_clusters,
    token_trigrams,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--rows-per-source", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=4201)
    args = parser.parse_args()
    if args.rows_per_source < 100:
        raise RuntimeError("audit source quota is too small")
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

    exact, index, exclusion_manifests = load_exclusions(args.exclude)
    candidates = fetch_videoufo(
        50_000,
        args.cache / "videoufo-pages",
        args.seed,
        cache_only=True,
    )
    candidates.extend(fetch_diffusiondb(args.cache, 120_000, args.seed + 1))
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in candidates:
        if not excluded_near_duplicate(row["prompt"], exact, index):
            grouped[row["source"]].append(row)
    rng = random.Random(args.seed)
    selected = []
    selected_exact = set(exact)
    selected_index = collections.defaultdict(list)
    for gram, values in index.items():
        selected_index[gram].extend(values)
    for source in ("VideoUFO", "DiffusionDB"):
        values = sorted(
            grouped[source], key=lambda row: (row["source_id"], row["prompt"])
        )
        rng.shuffle(values)
        count = 0
        for row in values:
            if excluded_near_duplicate(
                row["prompt"], selected_exact, selected_index
            ):
                continue
            selected.append(row)
            count += 1
            selected_exact.add(row["prompt"].casefold())
            grams = token_trigrams(row["prompt"])
            for gram in grams:
                selected_index[gram].append(grams)
            if count == args.rows_per_source:
                break
        if count != args.rows_per_source:
            raise RuntimeError(f"{source} yielded {count} audit rows")
    rng.shuffle(selected)
    near_duplicate_clusters(selected)
    for row_id, row in enumerate(selected):
        row["id"] = row_id
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "rows": len(selected),
        "rows_per_source": args.rows_per_source,
        "source_revisions": {
            "VideoUFO": VIDEOUFO_REVISION,
            "DiffusionDB": DIFFUSIONDB_REVISION,
        },
        "prior_exclusions": exclusion_manifests,
        "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "usage": "post-freeze independent audit sampling only",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
