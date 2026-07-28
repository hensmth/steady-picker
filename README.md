# SteadyPicker

SteadyPicker chooses quota-aware settings for video generation. It runs locally,
requires no production LLM or network call, and returns a duration, aspect
ratio, resolution, estimated cost, and cryptographic model provenance.

It is deliberately conservative. The `v0.2.0` model may extend a prompt from
the safe four-second default to six seconds when it detects a sufficiently
strong multi-stage action. It never uses a learned decision to shorten a video.

## What it does

For every request, SteadyPicker applies this order:

1. Honor valid explicit settings in the prompt or JSON request.
2. Preserve source-media constraints for image-to-video.
3. Allow a high-confidence learned `long` decision to select six seconds.
4. Use the safe four-second, 480p fallback for everything else.

Aspect ratio and resolution are deterministic; the model controls duration
only. The built-in `quota-safe-v2` profile behaves as follows:

| Field | Default | Allowed behavior |
| --- | --- | --- |
| Duration | 4 seconds | Explicit 2/4/6 seconds; learned decisions may select 6 |
| Resolution | 480p | Explicit 720p is allowed |
| Text-to-video aspect | 16:9 | Explicit 1:1, 16:9, or 9:16 |
| Image-to-video aspect | Source ratio | Preserved unless explicitly governed otherwise |
| Estimated 480p price | $0.05/second | $0.10, $0.20, or $0.30 for 2/4/6 seconds |
| Estimated 720p price | $0.07/second | $0.14, $0.28, or $0.42 for 2/4/6 seconds |

Examples:

| Request | Result | Why |
| --- | --- | --- |
| `render for 2 seconds` | 2 seconds | Valid explicit duration |
| `render for 2 seconds and 6 seconds` | 4 seconds | Conflicting cues fall back safely |
| A normal prompt with no accepted override | 4 seconds | Safe fallback |
| An accepted high-confidence multi-stage prompt | 6 seconds | Learned `long` decision |
| Image-to-video with no aspect cue | Source aspect | Avoid stretching the input |

Negated, unsupported, and conflicting technical cues are handled
deterministically. Unsupported durations round upward to the next allowed
duration, values above the profile maximum clamp, and ambiguous duration cues
fall back to four seconds.

## Measured model quality

These numbers describe the released `settings-v2` artifact and
`long-only-pragmatic-v2` decision policy. They are not a claim of 79.31%
accuracy over all prompts.

Correctness is measured against unanimous structured votes from three blind
Sol Ultra teacher passes. It is AI-adjudicated agreement, not human preference,
rendered-video quality, or a guarantee that every viewer would choose the same
duration.

| Evaluation | Accepted long decisions | Correct | Precision | Coverage / uncertainty |
| --- | ---: | ---: | ---: | --- |
| Five-fold grouped cross-validation | 29 | 23 | 79.31% | One-sided exact 95% lower bound ≈63.2% |
| Policy-development split | 8 of 2,000 | 6 | 75.00% | 0.40% override coverage |
| Sealed replacement holdout | 8 of 1,000 | 8 | 100.00% | 0.80% coverage; exact lower bound 68.77% |

“Precision” asks: **when SteadyPicker overrides the four-second fallback and
chooses six seconds, how often did the teacher label agree?** Most prompts are
not overridden, so the model's overall behavior is dominated by the safe
four-second fallback.

The sealed result is encouraging but statistically small. Eight correct
decisions do not establish 100% production accuracy. The release failed its
original minimum-support gate because it needed at least 20 accepted sealed
examples and produced only eight. The stricter original 98% long-precision
target also was not met during model selection.

The practical tradeoff is:

- approximately one unnecessary extension in five accepted cross-validation
  decisions;
- an accepted extension adds $0.10 at 480p or $0.14 at 720p versus the
  four-second baseline;
- learned shortening is disabled, eliminating model-caused
  under-duration; and
- unusual, multilingual, adversarial, or shifted prompts may perform worse.

Semantic inference measured 32.825 ms p95 on the benchmark machine, missing the
30 ms target by 2.825 ms. Go tests, race detection, vet, loader/profile/cue
fuzzing, and cross-platform builds passed.

See [MODEL_CARD.md](MODEL_CARD.md) and
[evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json](evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json)
for the complete evidence and failed-gate disclosure.

## Quick start

Install the exact released version:

```bash
go install github.com/hensmth/steady-picker/cmd/steady-picker@v0.2.0
```

Or download a ready-to-run Linux, macOS, or Windows binary from the
[v0.2.0 release](https://github.com/hensmth/steady-picker/releases/tag/v0.2.0).
Verify downloaded assets with `SHA256SUMS` and GitHub artifact attestations.

Run one JSON request per line through stdin:

```bash
printf '%s\n' \
  '{"prompt":"a flower progresses from bud to full bloom","mode":"text-to-video"}' |
  steady-picker predict --profile quota-safe-v2
```

Example result:

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
  "artifact_sha256": "2ea8edf030b351cdcd3e992ff1c15fbbaab3b49bb0b64fa04b506db318eba5a3",
  "reasons": [
    "duration.safe_fallback",
    "aspect.safe_fallback",
    "resolution.safe_fallback"
  ],
  "estimated_cost_microusd": 200000,
  "pricing_as_of": "2026-07-28"
}
```

Prompts are accepted exclusively through stdin and do not appear in process
arguments.

## Input and output behavior

```json
{
  "prompt": "one quick wink in a portrait frame",
  "mode": "text-to-video",
  "duration": 2,
  "aspect_ratio": "9:16",
  "resolution": "480p"
}
```

- `mode` must be `text-to-video` or `image-to-video`.
- Prompts must be valid UTF-8, non-empty, and no larger than 16 KiB.
- Image requests may provide `source_media_aspect_ratio`.
- The v0.1 `image_aspect_ratio` number remains accepted for compatibility.
- `source` remains a compatibility alias for `duration_source`.
- `confidence` is nonzero only for an accepted learned decision.
- `reasons` are stable machine-readable decision explanations.
- `artifact_sha256` identifies the model bytes actually loaded.
- Cost values use micro-USD so callers do not need floating-point currency.

Use a custom governed profile:

```bash
steady-picker predict --profile-file ./profile.json < requests.jsonl
```

A profile defines allowed durations, semantic mappings, resolutions, aspect
ratios, defaults, maximums, and optional per-second price estimates. Provider
capabilities and application budgets therefore remain separate from the model.

## Model and training

The embedded model is a deterministic, quantized semantic encoder:

- two Transformer layers;
- hidden size 128 and four attention heads;
- FFN size 512;
- 8,192-token WordPiece vocabulary;
- deterministic 96-token maximum;
- separate `short` and `long` binary heads; and
- auxiliary heads for action beats, ordered progression, transformation, and
  dependent actions.

Runtime inference is pure Go with `CGO_ENABLED=0`, pooled workspaces, no ONNX
runtime, no Python, no external model file, and no network access.

Training used a governed 15,000-prompt corpus:

- 14,000 development prompts;
- a separate 1,000-prompt sealed replacement holdout;
- rights-cleared VideoUFO CC-BY-4.0 captions and DiffusionDB CC0 prompts;
- three blind structured teacher votes per prompt; and
- unanimous labels only for learned overrides.

Disagreements expect the safe fallback. Near-duplicate clusters stay together
across splits. Model selection used five-fold grouped cross-validation;
temperature, conformal values, and policy thresholds were frozen before the
sealed holdout was opened.

The exact training and evaluation process is documented in
[docs/SEMANTIC_RETRAINING.md](docs/SEMANTIC_RETRAINING.md) and
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Operations and provenance

```bash
steady-picker health
steady-picker inspect-model
steady-picker licenses
steady-picker predict --model ./custom-v5.bin < requests.jsonl
```

The released health response identifies:

```json
{
  "status": "ok",
  "ready": true,
  "model_version": "settings-v2",
  "artifact_format": 5,
  "policy_version": "quota-safe-v2",
  "decision_policy": "long-only-pragmatic-v2",
  "artifact_sha256": "2ea8edf030b351cdcd3e992ff1c15fbbaab3b49bb0b64fa04b506db318eba5a3"
}
```

Artifact formats v3, v4, and v5 are supported. Models over 64 MiB, malformed
dimensions, non-finite values, unknown labels, or ambiguous legacy v2 label
order are rejected before inference. Use SteadyPicker v0.1 for a legacy v2
artifact.

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
        Mode:   steady.TextToVideo,
    },
)
if err != nil {
    log.Fatal(err)
}
```

Public entry points are `Load`, `LoadBytes`, `LoadDefault`, `Model.Metadata`,
`NewProfile`, `QuotaSafeProfile`, and `PickSettings`. Loaded models are
immutable and concurrency-safe; no `Close` call is required. Returned
classification slices are caller-owned.

## Development

```bash
go test ./...
go test -race ./...
go vet ./...
govulncheck ./...
go test -run '^$' -bench BenchmarkPickSettingsFallback -benchmem
```

CI runs tests and cross-platform builds on Linux, macOS, and Windows, plus
CodeQL and dependency review.

## License

Source code is MIT. The published v2 corpus and model are CC-BY-4.0.
VideoUFO attribution and DiffusionDB/FETV notices are embedded and available
through `steady-picker licenses`. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
