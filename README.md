# SteadyPicker

SteadyPicker is a local, deterministic video-settings engine. It chooses a
quota-aware duration and parses aspect ratio and resolution without making a
network or LLM call in production.

The CLI embeds an artifact, attribution, and the `quota-safe-v2` profile. A
single executable is enough.

> **v0.2 status:** model selection failed the frozen long-override gate. The
> branch therefore embeds a release-blocked bootstrap artifact that always
> leaves learned duration at the safe fallback. `steady-picker health` reports
> `"status":"bootstrap"` and `"ready":false`. Use v0.1 for the released model;
> see `evaluation/README.md` for the complete v0.2 evidence.

## Quick start

```bash
go install github.com/hensmth/steady-picker/cmd/steady-picker@latest

printf '%s\n' \
  '{"prompt":"a flower progresses from bud to full bloom","mode":"text-to-video"}' |
  steady-picker predict --profile quota-safe-v2
```

`predict` accepts newline-delimited JSON exclusively through stdin, so prompts
do not appear in process arguments. It emits one result per input line:

```json
{
  "duration": 4,
  "aspect_ratio": "16:9",
  "resolution": "480p",
  "source": "fallback",
  "duration_source": "fallback",
  "aspect_ratio_source": "fallback",
  "resolution_source": "fallback",
  "confidence": 0,
  "model_version": "settings-v2",
  "policy_version": "quota-safe-v2",
  "profile_version": "2",
  "artifact_sha256": "...",
  "reasons": ["duration.safe_fallback", "aspect.safe_fallback", "resolution.safe_fallback"],
  "estimated_cost_microusd": 200000,
  "pricing_as_of": "2026-07-28"
}
```

Ready-to-run Linux, macOS, and Windows binaries are attached to each release.
Verify them with `SHA256SUMS` and GitHub artifact attestations before use.

## Policy profiles

Provider capabilities and application budgets are separate. A `Profile`
defines allowed durations, semantic mappings, resolutions, aspect ratios,
defaults, maximums, and optional per-second price estimates.

The embedded `quota-safe-v2` profile:

- maps learned `short`, `medium`, and `long` labels to 2, 4, and 6 seconds;
- defaults to 4 seconds, 480p, and 16:9 for text-to-video;
- permits an explicit 720p request;
- preserves source aspect ratio for image-to-video by default; and
- estimates 480p at 50,000 micro-USD/sec and 720p at 70,000 micro-USD/sec,
  with a dated pricing field.

Decision precedence is field-specific: valid explicit request, source-media
constraint, accepted learned duration, then safe fallback. The model never
controls aspect ratio or resolution.

Use a custom governed profile:

```bash
steady-picker predict --profile-file ./profile.json < requests.jsonl
```

## Input

```json
{
  "prompt": "one quick wink in a portrait frame",
  "mode": "text-to-video",
  "duration": 2,
  "aspect_ratio": "9:16",
  "resolution": "480p"
}
```

`mode` is `text-to-video` or `image-to-video`. Image requests may provide
`source_media_aspect_ratio`; the v0.1 `image_aspect_ratio` number remains
accepted for compatibility. Prompts must be valid UTF-8, non-empty, and no
larger than 16 KiB.

Technical prompt cues are negation- and conflict-aware. Unsupported durations
round upward to the next profile duration and values above the maximum clamp.
Conflicting affirmative cues use the safe fallback.

## Operations

```bash
steady-picker health
steady-picker inspect-model
steady-picker licenses
steady-picker predict --model ./custom-v3.bin < requests.jsonl
```

Artifact v3 is strict and provenance-carrying. Models over 64 MiB, malformed
dimensions, non-finite values, unknown labels, or ambiguous v2 label order are
rejected before inference. Use v0.1 for a legacy v2 artifact or retrain it.

## Go library

```go
model, err := steady.LoadDefault()
if err != nil {
    log.Fatal(err)
}
result, err := steady.PickSettings(
    model,
    steady.QuotaSafeProfile(),
    steady.PickRequest{
        Prompt: "a person walks naturally across a room",
        Mode: steady.TextToVideo,
    },
)
```

Public entry points are `Load`, `LoadBytes`, `LoadDefault`,
`Model.Metadata`, `NewProfile`, `QuotaSafeProfile`, and `PickSettings`.
Loaded models are immutable and safe for concurrent use; no `Close` call is
needed. Returned classification slices are caller-owned.

## Reproducible training

The governed v2 pipeline uses exactly 3,000 VideoUFO brief captions and 2,000
DiffusionDB prompts. It excludes URLs, identities, technical settings, unsafe
rows, duplicates, and near-duplicate clusters before teacher labelling.

```bash
python scripts/build_corpus.py \
  --cache .cache --output corpus/prompts.jsonl \
  --extras-output corpus/audit-extras.jsonl \
  --manifest corpus/source-manifest.json

python scripts/label_with_hermes.py \
  --input corpus/prompts.jsonl \
  --output corpus/labelled.jsonl \
  --manifest corpus/teacher-manifest.json \
  --checkpoint-dir .checkpoints \
  --provider openai-codex --teacher-model gpt-5.6-sol

python scripts/build_splits.py \
  --input corpus/labelled.jsonl \
  --output-dir corpus/splits --manifest corpus/splits-manifest.json
```

Teacher labelling is local by default and supports optional generic SSH
execution. It runs two blind shuffled passes with permuted label presentation
and a third disagreement pass. Checkpoints are validated before resume.

Training requires four already-frozen splits:

```bash
steady-picker train \
  --train corpus/splits/train.txt \
  --probability-calibration corpus/splits/probability_calibration.txt \
  --conformal-calibration corpus/splits/conformal_calibration.txt \
  --threshold-development corpus/splits/policy_development.txt \
  --source-manifest-sha256 SHA256 \
  --training-code-commit GIT_SHA \
  --output models/settings-v2.bin
```

The locked test, FETV evaluation, and independent AI-adjudicated audit are
opened only after temperature, conformal quantiles, and policy thresholds are
frozen. See [MODEL_CARD.md](MODEL_CARD.md), the corpus dataset card, and
[the full methodology](docs/METHODOLOGY.md).

## Development

```bash
go test ./...
go test -race ./...
go vet ./...
govulncheck ./...
go test -run '^$' -bench BenchmarkPickSettingsFallback -benchmem
```

CI also runs CodeQL and dependency review on Linux, macOS, and Windows. Release
automation creates a draft only after the checked-in release marker proves all
locked gates passed.

## License

Source code is MIT. The published v2 corpus and model are CC-BY-4.0.
VideoUFO attribution and DiffusionDB/FETV notices are embedded and available
through `steady-picker licenses`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
