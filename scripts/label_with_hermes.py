#!/usr/bin/env python3
"""Run two blind Hermes teacher passes plus a disagreement adjudication pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import os
import time
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUBRIC = """Choose the minimum normal-paced clip needed to depict every requested
action without inventing filler. LABELS: {labels}.
short = one simple beat.
medium = one developed action or modest motion.
long = ordered stages, transformation, progression, or multiple dependent actions.
Return ONLY a JSON array in input order. Each item is {{"id": integer, "label": string}}.
Never return explanations or prompt text."""


def invoke(
    batch: list[dict],
    labels: list[str],
    args: argparse.Namespace,
    invocation_id: str,
) -> list[dict]:
    prompt = RUBRIC.format(labels=",".join(labels)) + "\nINPUT:\n" + json.dumps(batch)
    usage_file = args.usage_dir / f"{invocation_id}.json"
    command = [
        args.hermes,
        "--usage-file",
        str(usage_file),
        "--provider",
        args.provider,
        "-m",
        args.teacher_model,
        "--ignore-rules",
        "-z",
        prompt,
    ]
    invocation = ["ssh", args.ssh_host, shlex.join(command)] if args.ssh_host else command
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            remaining = int(args.deadline - time.monotonic())
            if remaining <= 0:
                raise RuntimeError("teacher wall-clock deadline expired")
            env = os.environ.copy()
            env["HERMES_HOME"] = str(args.hermes_home)
            if args.hermes_env:
                env["HERMES_ENV"] = str(args.hermes_env)
            result = subprocess.run(
                invocation,
                check=True,
                capture_output=True,
                text=True,
                timeout=min(args.timeout, remaining),
                env=env,
            )
            if args.require_provider_evidence:
                usage = json.loads(usage_file.read_text(encoding="utf-8"))
                evidence = json.dumps(usage, sort_keys=True).lower()
                for required, description in (
                    (args.provider, "provider"),
                    (args.teacher_model, "model"),
                ):
                    if required.lower() not in evidence:
                        raise RuntimeError(
                            f"teacher usage did not confirm requested {description}"
                        )
            start, end = result.stdout.find("["), result.stdout.rfind("]")
            if start < 0 or end < start:
                raise RuntimeError("teacher response did not contain a JSON array")
            labelled = json.loads(result.stdout[start : end + 1])
            if len(labelled) != len(batch):
                raise RuntimeError("teacher returned the wrong row count")
            for expected, received in zip(batch, labelled):
                if received.get("id") != expected["id"]:
                    raise RuntimeError("teacher changed row order or id")
                if received.get("label") not in {"short", "medium", "long"}:
                    raise RuntimeError(f"invalid teacher label: {received!r}")
            return labelled
        except (json.JSONDecodeError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = error
            if attempt == 2:
                break
    raise RuntimeError("teacher failed three validation attempts") from last_error


def pass_votes(
    rows: list[dict],
    pass_number: int,
    args: argparse.Namespace,
    checkpoint: Path,
) -> dict[int, str]:
    rng = random.Random(args.seed + pass_number)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    labels = ["short", "medium", "long"]
    rng.shuffle(labels)
    votes: dict[int, str] = {}
    if checkpoint.exists():
        loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
        votes = {int(key): value for key, value in loaded.items()}
        valid_ids = {int(row["id"]) for row in rows}
        if not set(votes) <= valid_ids or any(
            value not in {"short", "medium", "long"} for value in votes.values()
        ):
            raise RuntimeError(f"invalid checkpoint {checkpoint}")
    pending = [row for row in shuffled if int(row["id"]) not in votes]
    batches = [
        (
            offset // args.batch_size,
            [{"id": row["id"], "prompt": row["prompt"]} for row in pending[offset : offset + args.batch_size]],
        )
        for offset in range(0, len(pending), args.batch_size)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(batches), args.workers):
            wave = batches[offset : offset + args.workers]
            futures = [
                executor.submit(
                    invoke,
                    batch,
                    labels,
                    args,
                    f"pass-{pass_number}-batch-{batch_index:04d}",
                )
                for batch_index, batch in wave
            ]
            try:
                for future in futures:
                    for labelled in future.result():
                        votes[int(labelled["id"])] = labelled["label"]
                    temporary = checkpoint.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(votes, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    temporary.replace(checkpoint)
                    print(f"pass {pass_number}: {len(votes)}/{len(rows)}", flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    if set(votes) != {int(row["id"]) for row in rows}:
        raise RuntimeError("teacher pass did not preserve the exact input id set")
    return votes


def weighted_kappa(first: list[str], second: list[str]) -> float:
    order = {"short": 0, "medium": 1, "long": 2}
    n = len(first)
    observed = sum((order[a] - order[b]) ** 2 / 4 for a, b in zip(first, second)) / n
    frequencies = []
    for values in (first, second):
        frequencies.append([values.count(label) / n for label in order])
    expected = sum(
        frequencies[0][i] * frequencies[1][j] * ((i - j) ** 2 / 4)
        for i in range(3)
        for j in range(3)
    )
    return 1 - observed / expected if expected else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssh-host")
    parser.add_argument("--hermes", default="hermes")
    parser.add_argument("--provider", default="openai-codex")
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--deadline-seconds", type=int, default=3600)
    parser.add_argument("--reasoning-effort", default="ultra")
    parser.add_argument("--hermes-env", type=Path)
    parser.add_argument(
        "--hermes-auth",
        type=Path,
        default=Path.home() / ".hermes" / "auth.json",
    )
    parser.add_argument("--usage-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=5000)
    parser.add_argument("--require-provider-evidence", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise RuntimeError("--workers must be between 1 and 4")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    if len(rows) != args.expected_count or [row["id"] for row in rows] != list(range(args.expected_count)):
        raise RuntimeError(
            f"input must contain exactly ordered ids 0..{args.expected_count - 1}"
        )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.usage_dir.mkdir(parents=True, exist_ok=True)
    args.hermes_temp = tempfile.TemporaryDirectory(prefix="steady-hermes-")
    args.hermes_home = Path(args.hermes_temp.name)
    args.isolated_config = args.hermes_home / "config.yaml"
    args.isolated_config.write_text(
        json.dumps(
            {
                "agent": {"reasoning_effort": args.reasoning_effort},
                "model": {
                    "provider": args.provider,
                    "default": args.teacher_model,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not args.hermes_auth.is_file():
        raise RuntimeError("Hermes auth file is missing")
    shutil.copyfile(args.hermes_auth, args.hermes_home / "auth.json")
    os.chmod(args.hermes_home / "auth.json", 0o600)
    verification_env = os.environ.copy()
    verification_env["HERMES_HOME"] = str(args.hermes_home)
    verified_effort = subprocess.run(
        [args.hermes, "config", "get", "agent.reasoning_effort"],
        check=True,
        capture_output=True,
        text=True,
        env=verification_env,
    ).stdout.strip()
    if verified_effort != args.reasoning_effort:
        raise RuntimeError(
            f"isolated Hermes reasoning mismatch: {verified_effort!r}"
        )
    args.deadline = time.monotonic() + args.deadline_seconds

    votes: list[dict[int, str]] = []
    for pass_number in (1, 2):
        checkpoint = args.checkpoint_dir / f"teacher-pass-{pass_number}.json"
        vote = pass_votes(rows, pass_number, args, checkpoint)
        votes.append(vote)

    disagreements = [row for row in rows if votes[0][row["id"]] != votes[1][row["id"]]]
    third = (
        pass_votes(
            disagreements,
            3,
            args,
            args.checkpoint_dir / "teacher-pass-3.json",
        )
        if disagreements
        else {}
    )
    output: list[dict] = []
    unresolved = 0
    for row in rows:
        row_id = row["id"]
        row_votes = [votes[0][row_id], votes[1][row_id]]
        if row_id in third:
            row_votes.append(third[row_id])
        counts = {label: row_votes.count(label) for label in {"short", "medium", "long"}}
        final = max(sorted(counts), key=counts.get)
        is_unresolved = max(counts.values()) < 2
        unresolved += is_unresolved
        output.append(
            {
                **row,
                "teacher_votes": row_votes,
                "final_label": None if is_unresolved else final,
                "unresolved": is_unresolved,
            }
        )
    kappa = weighted_kappa(
        [votes[0][row["id"]] for row in rows],
        [votes[1][row["id"]] for row in rows],
    )
    if kappa < 0.80 or unresolved / len(rows) > 0.02:
        raise RuntimeError(f"labelling gate failed: kappa={kappa:.4f}, unresolved={unresolved}")
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8")
    usage_rows = []
    for path in sorted(args.usage_dir.glob("*.json")):
        try:
            usage_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            raise RuntimeError(f"invalid Hermes usage evidence: {path.name}")
    manifest = {
        "schema": 1,
        "provider": args.provider,
        "model": args.teacher_model,
        "reasoning_effort": args.reasoning_effort,
        "hermes": args.hermes,
        "ssh": bool(args.ssh_host),
        "seed": args.seed,
        "weighted_kappa": kappa,
        "unresolved": unresolved,
        "labelled_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "reasoning_recorded": True,
        "wall_clock_budget_seconds": args.deadline_seconds,
        "usage": {
            "files": len(usage_rows),
            "api_calls": sum(int(row.get("api_calls", 0)) for row in usage_rows),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in usage_rows),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in usage_rows),
            "reasoning_tokens": sum(int(row.get("reasoning_tokens", 0)) for row in usage_rows),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
