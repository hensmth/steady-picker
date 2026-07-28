# Semantic retraining runbook

This runbook is the public, machine-neutral command sequence for the v5
candidate. Paths below are illustrative working directories; manifests contain
the exact source revisions, input hashes, exclusion hashes, and tested commit.

Install the pinned training environment:

```bash
python -m venv .semantic-venv
.semantic-venv/bin/pip install -r requirements-semantic.lock
```

Build the governed 14,000-row development corpus plus the separate 1,000-row
sealed holdout. Every prior corpus, pilot, audit, and inspected test file must be
passed as an additional `--exclude` argument:

```bash
python scripts/build_semantic_corpus.py \
  --existing-labelled governed-existing-labelled.jsonl \
  --existing-membership governed-existing-membership.jsonl \
  --sealed-holdout-prompts sealed-holdout-prompts.jsonl \
  --cache .cache \
  --development-output work/development-prompts.jsonl \
  --fresh-output work/fresh-prompts.jsonl \
  --manifest work/source-manifest.json \
  --seed 20260729 \
  --videoufo-pool 50000 \
  --videoufo-cache-only \
  --exclude prior-corpus.jsonl
```

Start one external two-hour wall-clock deadline, then run three complete blind
structured passes over development. Pass only the remaining budget to each
invocation; the values below reserve ten minutes for the sealed support pass:

```bash
python scripts/label_with_hermes.py \
  --input work/development-prompts.jsonl \
  --output work/development-labelled.jsonl \
  --manifest work/development-teacher-manifest.json \
  --checkpoint-dir work/development-checkpoints \
  --usage-dir work/development-usage \
  --batch-size 200 --workers 4 \
  --provider openai-codex \
  --teacher-model gpt-5.6-sol \
  --reasoning-effort ultra \
  --deadline-seconds 6600 \
  --expected-count 14000 \
  --require-provider-evidence \
  --structured-unanimous
```

Label the sealed holdout separately. `--sealed-output` prevents label-derived
aggregate statistics from appearing before the support-only gate:

```bash
python scripts/label_with_hermes.py \
  --input sealed-holdout-prompts.jsonl \
  --output work/locked-labelled.sealed.jsonl \
  --manifest work/locked-teacher-manifest.sealed.json \
  --checkpoint-dir work/locked-checkpoints \
  --usage-dir work/locked-usage \
  --batch-size 200 --workers 4 \
  --provider openai-codex \
  --teacher-model gpt-5.6-sol \
  --reasoning-effort ultra \
  --deadline-seconds 600 \
  --expected-count 1000 \
  --require-provider-evidence \
  --structured-unanimous \
  --sealed-output

python scripts/validate_sealed_support.py \
  --prompts sealed-holdout-prompts.jsonl \
  --labels work/locked-labelled.sealed.jsonl \
  --output work/sealed-support.json
```

Stop before splitting or model selection unless the support validator reports
`row_integrity: true` and `unanimous_override_support_sufficient: true`. The
validator never emits class counts or examples.

If it fails, retire that holdout without inspecting counts. A single
prospectively specified support-enriched replacement may be selected from the
pinned source pools, with the development corpus, failed holdout, prior
corpora, audits, and external evaluation rows all supplied as exclusions:

```bash
python scripts/select_semantic_holdout.py \
  --cache .cache \
  --output work/replacement-locked-prompts.jsonl \
  --manifest work/replacement-locked-selection.json \
  --seed 20260730 \
  --videoufo-pool 120000 \
  --videoufo-cache-only \
  --exclude work/development-prompts.jsonl \
  --exclude work/failed-locked-prompts.jsonl \
  --exclude prior-corpus.jsonl
```

Run the same three sealed votes and support-only validator on that replacement.
Do not repeatedly resample holdouts or select from teacher labels.

Only after that gate passes, build splits and run the fixed 16-candidate,
five-fold selection. The training script does not accept a holdout-label path:

```bash
python scripts/build_semantic_splits.py \
  --input work/development-labelled.jsonl \
  --output-dir work/splits \
  --manifest work/splits-manifest.json \
  --seed 42

python scripts/train_semantic_v5.py \
  --development-labels work/development-labelled.jsonl \
  --split-dir work/splits \
  --source-manifest work/source-manifest.json \
  --teacher-manifest work/development-teacher-manifest.json \
  --sealed-support-validation work/sealed-support.json \
  --output-dir work/semantic-training \
  --artifact work/settings-v2.bin \
  --training-code-commit "$(git rev-parse HEAD)" \
  --decision-policy long-only-pragmatic-v2 \
  --recompute-cv
```

The pragmatic run reproduces frozen candidate 8 only. It must match pooled
evidence 23/29 and folds 4/6, 5/6, 3/4, 6/8, and 5/5 exactly; otherwise it
stops while the holdout remains sealed. Learned short decisions are disabled.
The final long threshold maximizes policy-development coverage subject to 75%
precision. If selection passes, run the locked, FETV, and blinded audit using
`--release-policy long-only-pragmatic-v2`. Publication still requires
`evaluation/LONG_ONLY_PRAGMATIC_V2_PASSED`.

Engineering gates:

```bash
go test ./...
go test -race ./...
go vet ./...
govulncheck ./...
go test -run '^$' -fuzz '^FuzzLoadBytes$' -fuzztime=30s
go test -run '^$' -fuzz '^FuzzProfileJSON$' -fuzztime=30s
go test -run '^$' -fuzz '^FuzzCueParsing$' -fuzztime=30s

STEADY_BENCH_MODEL=work/settings-v2.bin \
STEADY_BENCH_PROMPTS=work/development-prompts.jsonl \
go test -run '^TestExternalSemanticV5P95Latency$' -v
```
