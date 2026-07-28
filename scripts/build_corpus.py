#!/usr/bin/env python3
"""Build the deterministic, rights-cleared 5,000-prompt settings-v2 corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path

import pyarrow.parquet as pq

VIDEOUFO_REVISION = "683425832ad5b35d57d213963ecec18d5b689652"
DIFFUSIONDB_REVISION = "fb620fbe49fa4420e0734bd9c0df11f51176b61f"
VIDEOUFO_ROWS = "https://datasets-server.huggingface.co/rows"
DIFFUSIONDB = (
    "https://huggingface.co/datasets/poloclub/diffusiondb/"
    f"resolve/{DIFFUSIONDB_REVISION}/metadata.parquet"
)

URL_OR_IDENTITY = re.compile(
    r"(?:https?://|www\.|\b[\w.+-]+@[\w.-]+\.\w+\b|(?<!\w)@\w+)",
    re.IGNORECASE,
)
TECHNICAL = re.compile(
    r"\b(?:[1-9]|1[0-5])\s*(?:s|sec|second)s?\b|"
    r"\b(?:480p|720p|1080p|[248]\s*k)\b|"
    r"\b\d{2,5}\s*[x×]\s*\d{2,5}\s*(?:px|pixels?)?\b|"
    r"\b(?:fps|frames?\s+per\s+second)\b|"
    r"\b(?:1\s*:\s*1|16\s*:\s*9|9\s*:\s*16|4\s*:\s*3|3\s*:\s*4)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"[a-z0-9]+")
TEMPORAL_CUE = re.compile(
    r"\b(?:then|before|after|finally|first|next|gradually|eventually|"
    r"transform(?:s|ed|ing|ation)?|evolv(?:es|ed|ing)|"
    r"turn(?:s|ed|ing)?\s+into|time[- ]?lapse|journey|through|"
    r"different (?:types|places|planets|scenes)|"
    r"various (?:scenes|rooms|events)|multiple scenes|"
    r"from\s+\w+(?:\s+\w+){0,3}\s+to)\b",
    re.IGNORECASE,
)
ACTION_CUE = re.compile(
    r"\b(?:pour(?:s|ing)?|add(?:s|ing)?|mix(?:es|ing)?|cut(?:s|ting)?|"
    r"open(?:s|ing)?|shoot(?:s|ing)?|fight(?:s|ing)?|attack(?:s|ing)?|"
    r"driv(?:es|ing)|walk(?:s|ing)?|run(?:s|ning)?|travel(?:s|ing)?|"
    r"play(?:s|ing)?|cook(?:s|ing)?|paint(?:s|ing)?|sew(?:s|ing)?|"
    r"shape(?:s|ing)?|split(?:s|ting)?|explod(?:es|ing)|form(?:s|ing)?|"
    r"pick(?:s|ed|ing)?|tak(?:es|en|ing)|put(?:s|ting)?|discover(?:s|ed|ing)?|"
    r"emerg(?:es|ed|ing)|spread(?:s|ing)?|ris(?:es|en|ing)|"
    r"destroy(?:s|ed|ing)?|creat(?:es|ed|ing)|fold(?:s|ed|ing)?|"
    r"break(?:s|ing)?|crash(?:es|ed|ing)?|caus(?:es|ed|ing)|"
    r"squeez(?:es|ed|ing)|plac(?:es|ed|ing)|encounter(?:s|ed|ing)?|"
    r"forc(?:es|ed|ing)|rid(?:es|den|ing)|smok(?:es|ed|ing)|"
    r"drink(?:s|ing)?|shop(?:s|ped|ping)|approach(?:es|ed|ing)?|"
    r"pass(?:es|ed|ing)?|build(?:s|ing)?|construct(?:s|ed|ing)?|"
    r"drag(?:s|ged|ging)|spark(?:s|ed|ing)?|burst(?:s|ing)?|"
    r"dissolv(?:es|ed|ing)|swim(?:s|ming)?|fir(?:es|ed|ing)|"
    r"hold(?:s|ing)?|eat(?:s|en|ing)|mak(?:es|ing))\b",
    re.IGNORECASE,
)
DEPENDENT_ACTION = re.compile(
    r"\b(?:and then|and\s+(?:\w+\s+){0,3}(?:add|pour|mix|cut|shoot|"
    r"attack|driv|walk|run|play|cook|paint|open|form|pick|tak|put|"
    r"discover|emerg|spread|ris|destroy|creat|fold|break|crash|caus|"
    r"squeez|plac|encounter|forc|rid|smok|drink|shop|approach|pass|"
    r"build|construct|drag|spark|burst|dissolv|swim|fir|hold|eat|mak)\w*)\b",
    re.IGNORECASE,
)


def normalize(value: object) -> str | None:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split())
    if not 12 <= len(text) <= 1000:
        return None
    if URL_OR_IDENTITY.search(text) or TECHNICAL.search(text):
        return None
    printable = sum(character.isprintable() for character in text)
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    letters = sum(character.isalpha() for character in text)
    if printable != len(text) or letters < 4 or ascii_letters / max(letters, 1) < 0.85:
        return None
    return text


def token_trigrams(text: str) -> frozenset[tuple[str, str, str]]:
    tokens = WORD.findall(text.lower())
    return frozenset(zip(tokens, tokens[1:], tokens[2:]))


def load_exclusions(paths: list[Path]) -> tuple[set[str], dict[tuple[str, str, str], list[frozenset]], list[dict]]:
    exact: set[str] = set()
    index: dict[tuple[str, str, str], list[frozenset]] = collections.defaultdict(list)
    manifests: list[dict] = []
    for path in paths:
        digest = file_sha256(path)
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = normalize(row.get("prompt"))
            if not prompt:
                continue
            exact.add(prompt.casefold())
            grams = token_trigrams(prompt)
            for gram in grams:
                index[gram].append(grams)
            count += 1
        manifests.append({"sha256": digest, "rows": count})
    return exact, index, manifests


def excluded_near_duplicate(
    prompt: str,
    exact: set[str],
    index: dict[tuple[str, str, str], list[frozenset]],
    threshold: float = 0.85,
) -> bool:
    if prompt.casefold() in exact:
        return True
    grams = token_trigrams(prompt)
    candidates: dict[frozenset, None] = {}
    for gram in grams:
        for other in index.get(gram, ()):
            candidates[other] = None
    for other in candidates:
        union = len(grams | other)
        if union and len(grams & other) / union >= threshold:
            return True
    return False


def temporal_complexity(text: str) -> int:
    """Score rubric-relevant progression cues without assigning a label."""
    return (
        2 * len(TEMPORAL_CUE.findall(text))
        + min(2, len(ACTION_CUE.findall(text)))
        + int(bool(DEPENDENT_ACTION.search(text)))
    )


def near_duplicate_clusters(rows: list[dict], threshold: float = 0.85) -> None:
    """Assign deterministic clusters using a trigram inverted index."""
    representatives: list[frozenset] = []
    index: dict[tuple[str, str, str], set[int]] = collections.defaultdict(set)
    for row in rows:
        grams = token_trigrams(row["prompt"])
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(index[gram])
        cluster = len(representatives)
        for candidate in sorted(candidates):
            other = representatives[candidate]
            union = len(grams | other)
            if union and len(grams & other) / union >= threshold:
                cluster = candidate
                break
        if cluster == len(representatives):
            representatives.append(grams)
            for gram in grams:
                index[gram].add(cluster)
        row["cluster_id"] = f"cluster-{cluster:05d}"


def fetch_videoufo(limit: int, cache: Path, seed: int, cache_only: bool = False) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    first_page = cache / "0000000.json"
    if not first_page.exists():
        query = urllib.parse.urlencode(
            {"dataset": "WenhaoWang/VideoUFO", "config": "default", "split": "Full", "offset": 0, "length": 100}
        )
        with urllib.request.urlopen(f"{VIDEOUFO_ROWS}?{query}", timeout=60) as response:
            first_page.write_text(json.dumps(json.load(response)), encoding="utf-8")
    total_rows = json.loads(first_page.read_text(encoding="utf-8"))["num_rows_total"]
    offsets = (
        [int(path.stem) for path in cache.glob("*.json")]
        if cache_only
        else list(range(0, total_rows, 100))
    )
    random.Random(seed).shuffle(offsets)
    if 0 in offsets:
        offsets.remove(0)
    offsets.insert(0, 0)
    for offset in offsets:
        if len(candidates) >= limit:
            break
        query = urllib.parse.urlencode(
            {
                "dataset": "WenhaoWang/VideoUFO",
                "config": "default",
                "split": "Full",
                "offset": offset,
                "length": 100,
            }
        )
        page = cache / f"{offset:07d}.json"
        if page.exists():
            payload = json.loads(page.read_text(encoding="utf-8"))
        else:
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(f"{VIDEOUFO_ROWS}?{query}", timeout=60) as response:
                        payload = json.load(response)
                    page.write_text(json.dumps(payload), encoding="utf-8")
                    time.sleep(0.4)
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 429 and not 500 <= error.code < 600:
                        raise
                    if attempt == 7:
                        raise
                    retry_after = int(error.headers.get("Retry-After", "0") or 0)
                    time.sleep(max(retry_after, 2 ** attempt))
        source_rows = payload.get("rows", [])
        if not source_rows:
            break
        for wrapped in source_rows:
            source = wrapped["row"]
            prompt = normalize(source.get("Brief_Caption"))
            if prompt:
                dynamic = float(source.get("Dynamic_Degree") or 0)
                candidates.append(
                    {
                        "prompt": prompt,
                        "source": "VideoUFO",
                        "source_id": str(source.get("ID", wrapped.get("row_idx"))),
                        "topic": str(source.get("Topic") or "unknown").strip().lower(),
                        "dynamic_degree": dynamic,
                        "dynamic_band": "high" if dynamic >= 0.5 else "low",
                        "temporal_complexity": temporal_complexity(prompt),
                    }
                )
    return candidates


def round_robin_strata(rows: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["topic"], row["dynamic_band"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda row: (row["source_id"], row["prompt"]))
        rng.shuffle(values)
        values.sort(key=lambda row: row["temporal_complexity"])
    selected: list[dict] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        else:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return selected


def select_stratified_videoufo(
    rows: list[dict], count: int, temporal_count: int, seed: int, temporal_min_score: int
) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = row["prompt"].casefold()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    temporal = round_robin_strata(
        [row for row in unique if row["temporal_complexity"] >= temporal_min_score],
        temporal_count,
        seed + 1,
    )
    if len(temporal) != temporal_count:
        raise RuntimeError(
            f"VideoUFO yielded {len(temporal)} temporal rows, need {temporal_count}; "
            "increase --videoufo-pool"
        )
    temporal_ids = {row["source_id"] for row in temporal}
    remaining = round_robin_strata(
        [row for row in unique if row["source_id"] not in temporal_ids],
        count - temporal_count,
        seed + 2,
    )
    selected = temporal + remaining
    if len(selected) != count:
        raise RuntimeError(f"VideoUFO yielded {len(selected)} usable rows, need {count}")
    random.Random(seed + 3).shuffle(selected)
    return selected


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_diffusiondb(cache: Path, count: int, seed: int) -> list[dict]:
    parquet = cache / "diffusiondb.parquet"
    download(DIFFUSIONDB, parquet)
    table = pq.read_table(parquet)
    names = set(table.column_names)
    prompt_column = next(name for name in ("prompt", "p", "text") if name in names)
    nsfw_column = next((name for name in ("image_nsfw", "nsfw", "nsfw_score") if name in names), None)
    prompts = table[prompt_column].to_pylist()
    nsfw_values = table[nsfw_column].to_pylist() if nsfw_column else [0] * len(prompts)
    rows: list[dict] = []
    seen: set[str] = set()
    for index, (raw_prompt, raw_nsfw) in enumerate(zip(prompts, nsfw_values)):
        prompt = normalize(raw_prompt)
        if not prompt:
            continue
        try:
            nsfw = float(raw_nsfw or 0)
        except (TypeError, ValueError):
            continue
        key = prompt.casefold()
        if nsfw > 0.1 or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "prompt": prompt,
                "source": "DiffusionDB",
                "source_id": str(index),
                "nsfw_score": nsfw,
                "temporal_complexity": temporal_complexity(prompt),
            }
        )
    random.Random(seed).shuffle(rows)
    if len(rows) < count:
        raise RuntimeError(f"DiffusionDB yielded {len(rows)} usable rows, need {count}")
    return rows[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extras-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--videoufo-pool", type=int, default=12_000)
    parser.add_argument("--videoufo-cache-only", action="store_true")
    parser.add_argument("--video-temporal", type=int, default=550)
    parser.add_argument("--diffusion-temporal", type=int, default=1450)
    parser.add_argument("--temporal-min-score", type=int, default=3)
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="prior JSONL corpus to exclude by exact and near-duplicate prompt",
    )
    args = parser.parse_args()

    with urllib.request.urlopen("https://huggingface.co/api/datasets/WenhaoWang/VideoUFO", timeout=30) as response:
        if json.load(response)["sha"] != VIDEOUFO_REVISION:
            raise RuntimeError("VideoUFO revision changed; review and pin before rebuilding")
    video_candidates = fetch_videoufo(
        args.videoufo_pool,
        args.cache / "videoufo-pages",
        args.seed,
        args.videoufo_cache_only,
    )
    excluded_exact, excluded_index, exclusion_manifests = load_exclusions(args.exclude)
    video_candidates = [
        row
        for row in video_candidates
        if not excluded_near_duplicate(
            row["prompt"], excluded_exact, excluded_index
        )
    ]
    video = select_stratified_videoufo(
        video_candidates,
        3000,
        args.video_temporal,
        args.seed,
        args.temporal_min_score,
    )
    diffusion_candidates = fetch_diffusiondb(args.cache, 120_000, args.seed)
    diffusion_candidates = [
        row
        for row in diffusion_candidates
        if not excluded_near_duplicate(
            row["prompt"], excluded_exact, excluded_index
        )
    ]
    video_prompts = {row["prompt"].casefold() for row in video}
    eligible_diffusion = [
        row for row in diffusion_candidates
        if row["prompt"].casefold() not in video_prompts
    ]
    temporal_diffusion = [
        row for row in eligible_diffusion
        if row["temporal_complexity"] >= args.temporal_min_score
    ]
    temporal_diffusion.sort(key=lambda row: row["temporal_complexity"], reverse=True)
    temporal_diffusion = temporal_diffusion[:args.diffusion_temporal]
    temporal_ids = {row["source_id"] for row in temporal_diffusion}
    diffusion = temporal_diffusion + [
        row for row in eligible_diffusion
        if row["source_id"] not in temporal_ids
    ][:2000 - len(temporal_diffusion)]
    if len(diffusion) != 2000:
        raise RuntimeError("DiffusionDB could not supply 2,000 cross-source unique prompts")
    if len(temporal_diffusion) != args.diffusion_temporal:
        raise RuntimeError(
            f"DiffusionDB yielded {len(temporal_diffusion)} temporal rows, "
            f"need {args.diffusion_temporal}"
        )
    random.Random(args.seed + 4).shuffle(diffusion)
    rows = video + diffusion
    exact: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        key = row["prompt"].casefold()
        if key not in exact:
            exact.add(key)
            deduped.append(row)
    if len(deduped) != 5000:
        raise RuntimeError(f"exact duplicate handling produced {len(deduped)} rows")
    near_duplicate_clusters(deduped)
    for index, row in enumerate(deduped):
        row["id"] = index

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in deduped)
    args.output.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode()).hexdigest()
    manifest = {
        "schema": 1,
        "seed": args.seed,
        "rows": 5000,
        "sources": {
            "VideoUFO": {"revision": VIDEOUFO_REVISION, "license": "CC-BY-4.0", "rows": 3000},
            "DiffusionDB": {
                "revision": DIFFUSIONDB_REVISION,
                "license": "CC0-1.0",
                "rows": 2000,
                "raw_parquet_sha256": file_sha256(args.cache / "diffusiondb.parquet"),
            },
        },
        "corpus_sha256": digest,
        "filters": {
            "near_duplicate": "normalized token 3-gram Jaccard >= 0.85",
            "technical_settings": "excluded",
            "identity_metadata": "URLs, emails, and handles excluded",
            "diffusiondb_nsfw_max": 0.1,
            "temporal_sampling": {
                "definition": (
                    f"rubric-derived cue score >= {args.temporal_min_score}; "
                    "not a teacher label"
                ),
                "rows": {
                    "VideoUFO": args.video_temporal,
                    "DiffusionDB": args.diffusion_temporal,
                },
            },
            "prior_corpus_exclusions": exclusion_manifests,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.extras_output:
        selected_ids = {(row["source"], row["source_id"]) for row in deduped}
        extras = [
            row
            for row in video_candidates + diffusion_candidates
            if (row["source"], row["source_id"]) not in selected_ids
        ]
        random.Random(args.seed + 1000).shuffle(extras)
        extras = extras[:500]
        if len(extras) != 500:
            raise RuntimeError("could not construct 500 additional audit prompts")
        for index, row in enumerate(extras):
            row["id"] = index
        args.extras_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in extras),
            encoding="utf-8",
        )
    print(json.dumps({"rows": len(deduped), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
