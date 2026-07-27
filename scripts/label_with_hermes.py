#!/usr/bin/env python3
"""Label public prompts with Hermes, locally or through an optional SSH host."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


SYSTEM = """You label video-generation prompts for a quota-aware classifier.
Return ONLY a JSON array in the same order as the inputs.
Each item must be {"id": integer, "label": string}.
Allowed labels are: d2,d4,d6.
Duration: 2 for one quick beat, 4 for one normal action, 6 for a sequence,
timelapse, transformation, or several distinct actions.
Do not copy prompt text into the output."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--ssh-host")
    parser.add_argument("--hermes", default="hermes")
    parser.add_argument("--provider", default="openai-codex")
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    completed: dict[int, str] = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            row = json.loads(line)
            completed[int(row["id"])] = row["label"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    allowed = {f"d{duration}" for duration in (2, 4, 6)}
    with args.output.open("a", encoding="utf-8") as handle:
        pending = [row for row in rows if int(row["id"]) not in completed]
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            request = SYSTEM + "\nINPUT:\n" + json.dumps(batch)
            command = [
                args.hermes,
                "--safe-mode",
                "--provider",
                args.provider,
                "--model",
                args.teacher_model,
                "--oneshot",
                request,
            ]
            invocation = (
                ["ssh", args.ssh_host, shlex.join(command)]
                if args.ssh_host
                else command
            )
            result = subprocess.run(
                invocation,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            labels = json.loads(result.stdout)
            if len(labels) != len(batch):
                raise RuntimeError("teacher returned the wrong row count")
            for expected, labelled in zip(batch, labels):
                if labelled.get("id") != expected["id"]:
                    raise RuntimeError("teacher changed row order or id")
                if labelled.get("label") not in allowed:
                    raise RuntimeError(f"invalid label: {labelled}")
                record = {
                    "id": expected["id"],
                    "prompt": expected["prompt"],
                    "label": labelled["label"],
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            print(f"labelled {min(offset + len(batch), len(pending))}/{len(pending)}")


if __name__ == "__main__":
    main()
