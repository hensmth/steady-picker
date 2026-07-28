# SteadyPicker

[![Release](https://img.shields.io/github/v/release/hensmth/steady-picker)](https://github.com/hensmth/steady-picker/releases/latest)
[![CI](https://github.com/hensmth/steady-picker/actions/workflows/ci.yml/badge.svg)](https://github.com/hensmth/steady-picker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)](https://github.com/hensmth/steady-picker/releases/latest)
[![Offline](https://img.shields.io/badge/inference-offline-success)](#how-decisions-work)

SteadyPicker chooses quota-aware settings for video generation. It combines
deterministic rules with a small local model and returns a duration, aspect
ratio, resolution, estimated cost, and cryptographic model provenance.

The `v0.2.0` model is conservative and experimental: it may extend a prompt
from the safe four-second default to six seconds, but it never uses a learned
decision to select less than the four-second fallback.

| | |
| --- | --- |
| **Task** | Video-generation setting selection |
| **Input** | Newline-delimited JSON containing a prompt and generation mode |
| **Output** | Duration, aspect ratio, resolution, cost estimate, reasons, and provenance |
| **Runtime** | Pure Go, `CGO_ENABLED=0`, offline, no external model file |
| **Embedded model** | `settings-v2`, artifact format 5 |
| **Default profile** | `quota-safe-v2` |
| **Learned behavior** | High-confidence `long` decisions may select six seconds |
| **Release status** | Experimental; the published candidate failed its minimum-support and latency gates |
| **Licensing** | MIT code; CC-BY-4.0 model and governed dataset |

## Quick start

Install the exact released version:

```bash
go install github.com/hensmth/steady-picker/cmd/steady-picker@v0.2.0
```

Or use a ready-to-run binary from the
[v0.2.0 release](https://github.com/hensmth/steady-picker/releases/tag/v0.2.0).
Verify downloaded assets with the release checksums and GitHub artifact
attestations.

Send one JSON request per line through standard input:

```bash
printf '%s\n' \
  '{"prompt":"a flower progresses from bud to full bloom","mode":"text-to-video"}' |
  steady-picker predict --profile quota-safe-v2
```

Representative output:

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

Prompts are read exclusively from standard input and do not appear in process
arguments.

## How decisions work

SteadyPicker applies field-specific precedence:

1. Honor valid explicit settings.
2. Preserve source-media constraints for image-to-video.
3. Accept a sufficiently confident learned `long` decision for duration.
4. Use the safe deterministic fallback.

The model controls duration only. Aspect ratio, resolution, explicit-setting
parsing, conflict handling, and fallback behavior are deterministic.

| Request | Duration | Decision source |
| --- | ---: | --- |
| `render for 2 seconds` | 2 seconds | Explicit setting |
| `render for 2 seconds and 6 seconds` | 4 seconds | Safe fallback after conflicting cues |
| Ordinary prompt without an accepted override | 4 seconds | Safe fallback |
| Accepted multi-stage prompt | 6 seconds | Learned `long` decision |

The built-in profile allows 2, 4, or 6 seconds; 480p or 720p; and 1:1, 16:9,
or 9:16 text-to-video output. Image-to-video preserves the source aspect ratio
by default. Unsupported durations round upward to an allowed value, durations
above the profile maximum clamp, and ambiguous cues fall back safely.

The profile's cost estimates are metadata dated **2026-07-28**, not permanent
provider prices:

| Resolution | Estimated price per output second | 2 / 4 / 6-second estimate |
| --- | ---: | ---: |
| 480p | $0.05 | $0.10 / $0.20 / $0.30 |
| 720p | $0.07 | $0.14 / $0.28 / $0.42 |

## Quality at a glance

These results evaluate learned six-second overrides against unanimous
structured labels from three blind Sol Ultra teacher passes. They are not
human judgements, rendered-video quality measurements, or an accuracy score
over all prompts.

| Evaluation | Accepted long decisions | Correct | Long-override precision | Coverage / uncertainty |
| --- | ---: | ---: | ---: | --- |
| Five-fold grouped cross-validation | 29 | 23 | **79.31%** | One-sided exact 95% lower bound ≈63.2% |
| Policy-development split | 8 of 2,000 | 6 | 75.00% | 0.40% override coverage |
| Sealed replacement holdout | **8 of 1,000** | **8** | **100.00%** | **0.80% coverage; exact lower bound 68.77%** |

The headline result is **79.31% long-override precision on 29 accepted
cross-validation decisions**. Precision asks how often the teacher label
agreed when SteadyPicker overrode four seconds and selected six. Coverage is
the fraction of all prompts receiving that learned override.

The sealed 8/8 result is too small to establish 100% production accuracy. The
candidate failed its frozen minimum-support gate because it accepted only 8
sealed examples instead of the required 20. It also measured 32.825 ms p95
against a 30 ms target. FETV and the independent audit were not run after the
terminal support failure.

Learned short decisions are disabled. Consequently, this policy never selects
less than the four-second fallback from a learned decision, although a valid
explicit two-second request is still honored. An accepted extension adds an
estimated $0.10 at 480p or $0.14 at 720p relative to fallback.

See the [full model card](MODEL_CARD.md) and
[machine-readable diagnostics](evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json)
before using the model in production.

## When to use it

SteadyPicker is suited to applications that:

- need offline, reproducible, quota-aware video settings;
- prefer a four-second fallback when the model is uncertain;
- can tolerate an occasional unnecessary two-second extension; and
- independently validate model provenance and enforce their own budget profile.

Do not use it as:

- a safety classifier, content moderator, or video-quality evaluator;
- a general-purpose semantic encoder;
- a billing authority or source of permanently current provider prices;
- evidence of human preference or objective duration correctness; or
- a guaranteed selector for multilingual, adversarial, or shifted prompts.

## Input, profiles, and operations

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
- `source` remains a compatibility alias for `duration_source`.
- `confidence` is nonzero only for an accepted learned decision.
- Cost values use micro-USD to avoid floating-point currency.

Use a custom governed profile or inspect the embedded artifact:

```bash
steady-picker predict --profile-file ./profile.json < requests.jsonl
steady-picker health
steady-picker inspect-model
steady-picker licenses
```

A profile defines permitted settings, defaults, maximums, semantic-duration
mappings, and optional price estimates. Invalid input, an unavailable model, or
a rejected decision should be handled by the consuming application with its
own safe fallback.

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

Loaded models are immutable and concurrency-safe; no `Close` call is required.
Public entry points include `Load`, `LoadBytes`, `LoadDefault`,
`Model.Metadata`, `NewProfile`, `QuotaSafeProfile`, and `PickSettings`.

## Documentation

- [Model card](MODEL_CARD.md): intended use, risks, training, evaluation, and provenance
- [Semantic dataset card](docs/SEMANTIC_DATASET_CARD.md): corpus composition, labels, governance, and limitations
- [Methodology](docs/METHODOLOGY.md): data, selection, calibration, and engineering design
- [Frozen diagnostics](evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json): released candidate results and failed gates
- [Evaluation status](evaluation/README.md): human-readable gate record
- [Release assets and checksums](https://github.com/hensmth/steady-picker/releases/tag/v0.2.0)

## Development and license

```bash
go test ./...
go test -race ./...
go vet ./...
govulncheck ./...
```

CI tests Linux, macOS, and Windows, with CodeQL and dependency review. Source
code is MIT. The published v2 model is CC-BY-4.0; VideoUFO attribution and
DiffusionDB/FETV notices are available through `steady-picker licenses` and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
