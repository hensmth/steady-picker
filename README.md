# SteadyPicker

SteadyPicker is a local, deterministic video-settings engine. It chooses a
quota-aware duration and parses aspect ratio and resolution without making a
network or LLM call in production.

The CLI embeds an artifact, attribution, and the `quota-safe-v2` profile. A
single executable is enough.

> **v0.2 status:** the original strict short-and-long release gate remains
> failed. A separately marked pragmatic candidate enables learned `long`
> decisions only and keeps every learned short decision at the safe fallback.
> Its exact grouped-CV evidence passed the predeclared 75% precision gate, but
> the frozen sealed evaluation accepted only 8 examples against a minimum of
> 20. All 8 were correct and no learned short decision occurred. The embedded
> bootstrap is therefore unchanged and no release is published. See
> `evaluation/README.md`.

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

- maps semantic `short`, `medium`, and `long` labels to 2, 4, and 6 seconds;
- defaults to 4 seconds, 480p, and 16:9 for text-to-video;
- permits an explicit 720p request;
- preserves source aspect ratio for image-to-video by default; and
- estimates 480p at 50,000 micro-USD/sec and 720p at 70,000 micro-USD/sec,
  with a dated pricing field.

Decision precedence is field-specific: valid explicit request, source-media
constraint, accepted learned duration, then safe fallback. The pragmatic
decision policy accepts only learned `long` decisions. Explicit 2-second
requests still work. The model never controls aspect ratio or resolution.

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

Artifact v5 contains the compact semantic model and decision-policy marker;
v3 and v4 remain loadable. Models over 64 MiB, malformed
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

The semantic pipeline uses a governed 14,000-row development corpus and a
separate 1,000-row sealed holdout. Three blind structured teacher votes label
each prompt; disagreement means safe fallback. Near-duplicate clusters remain
together across splits.

The student is a deterministic, quantized two-layer Transformer with separate
short and long heads. The frozen pragmatic candidate uses learning rate
`0.0007`, long-class weight `0.75`, contrastive weight `0.25`, and focal
strength `1.0`. The public runbook contains the exact corpus, labelling,
selection, evaluation, and engineering commands:
[docs/SEMANTIC_RETRAINING.md](docs/SEMANTIC_RETRAINING.md).

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
