# Model Card: SteadyPicker settings-v2

## Model details

| | |
| --- | --- |
| **Model ID** | `settings-v2` |
| **Release** | `v0.2.0` |
| **Artifact format** | 5 |
| **Artifact SHA-256** | `2ea8edf030b351cdcd3e992ff1c15fbbaab3b49bb0b64fa04b506db318eba5a3` |
| **Task** | Semantic video-duration selection |
| **Decision policy** | `long-only-pragmatic-v2` |
| **Compatible profile** | `quota-safe-v2` |
| **Runtime** | Pure Go, offline, `CGO_ENABLED=0` |
| **License** | CC-BY-4.0 |
| **Status** | Experimental; failed frozen support and latency gates |

SteadyPicker `settings-v2` is a compact, quantized semantic encoder used by a
quota-aware video-setting policy. It decides whether a prompt provides enough
evidence to extend the deterministic four-second fallback to six seconds.

The released policy suppresses every learned `short` decision. Explicit
duration requests remain deterministic, and the model does not control aspect
ratio or resolution.

## Uses

### Direct use

Use the model for local, reproducible duration selection when:

- four seconds is an acceptable safe fallback;
- a conservative six-second extension is useful for multi-stage actions;
- low learned-override coverage is acceptable;
- an occasional unnecessary extension and its added cost are tolerable; and
- the caller verifies the artifact and maintains its own budget profile.

### Downstream use

Applications may combine the model with deterministic parsing, media
constraints, cost estimation, provenance validation, and an application-level
fallback. Profiles should translate the provider-neutral semantic decision
into the settings permitted by that application.

### Out-of-scope use

The model is not intended for:

- content safety, moderation, policy enforcement, or abuse detection;
- video-quality prediction or prompt ranking;
- objective or human-preference duration labelling;
- billing, entitlement, or permanently current price calculation;
- general-purpose text embeddings; or
- high-stakes decisions or unvalidated multilingual domains.

## Risks, limitations, and recommendations

- Labels are unanimous decisions from three blind Sol Ultra teacher passes,
  not human annotations or rendered-video assessments.
- The headline result has only 29 accepted cross-validation decisions. Its
  uncertainty is material.
- The sealed evaluation accepted only 8 of 1,000 examples. Its 8/8 result must
  be read alongside 0.8% coverage and a 68.77% one-sided exact lower bound.
- The model missed the frozen minimum-support gate and semantic latency target.
- FETV and the independent audit were omitted after the terminal gate failure.
- Performance may degrade on multilingual, unusual, adversarial, or shifted
  prompts. No conformal guarantee is claimed under distribution shift.
- An incorrect learned `long` decision spends more quota. At the profile
  estimates dated 2026-07-28, each extension adds $0.10 at 480p or $0.14 at
  720p over the four-second fallback.

Consumers should keep the deterministic fallback, validate the exact artifact
hash and policy, monitor accepted override rates, and separately evaluate their
own prompt distribution. Do not interpret the results as “79% accuracy.”

## Training data

The semantic workflow governs 15,000 prompts:

- 4,000 non-locked prompts retained from earlier governed data;
- 10,000 fresh, duplicate-excluded prompts comprising 4,000 broad examples,
  4,000 high-temporal-complexity examples, and 2,000 temporal-cue hard
  negatives; and
- a separate 1,000-prompt replacement holdout.

Only VideoUFO CC-BY-4.0 captions and DiffusionDB CC0 prompts enter training.
FETV is reserved for external evaluation. Sources with non-commercial or
ambiguous upstream terms are excluded. Exact and near duplicates from previous
corpora, pilots, audits, and inspected tests are excluded.

The 14,000 development rows are grouped by normalized token 3-gram
near-duplicate clusters and split into 10,000 training, 1,000 probability
calibration, 1,000 conformal calibration, and 2,000 policy-development rows.
Clusters do not cross splits.

See the [semantic dataset card](docs/SEMANTIC_DATASET_CARD.md) and
[methodology](docs/METHODOLOGY.md) for governance and collection details.

## Training procedure

Three blind, shuffled Sol Ultra passes return a duration label and structured
temporal fields: action-beat count, ordered progression, transformation, and
dependent actions. Only three-of-three duration agreement produces a learned
`short`, `medium`, or `long` label. Disagreement is ambiguous and expects the
safe fallback.

The model family uses:

- an Apache-2.0 MiniLM teacher encoder adapted with SetFit-style contrastive
  learning;
- a purpose-built two-layer Transformer student;
- hidden size 128, four attention heads, and FFN size 512;
- an 8,192-token WordPiece vocabulary with a deterministic 96-token maximum;
- separate `short` and `long` binary heads;
- structured temporal auxiliary heads; and
- per-output-channel INT8 weights with float32 layer normalization.

Five-fold grouped cross-validation evaluated a fixed 16-candidate grid over
learning rate, long-class weight, contrastive-loss weight, and focal-loss
strength. The pragmatic candidate was frozen as grid candidate 8:

| Hyperparameter | Value |
| --- | ---: |
| Learning rate | 0.0007 |
| Long-class weight | 0.75 |
| Contrastive-loss weight | 0.25 |
| Focal-loss strength | 1.0 |

Temperature, conformal values, and the final long threshold were fitted on
separate development splits and frozen before opening the replacement holdout.
The runtime policy then disabled learned short decisions.

## Evaluation

### Evaluation context and metrics

Correctness means agreement with the unanimous teacher duration label. It does
not measure human preference or the quality of generated videos.

Long-override precision is:

```text
correct accepted learned-long decisions / all accepted learned-long decisions
```

Coverage is the fraction of all prompts receiving a learned-long override.
Confidence bounds are one-sided exact 95% binomial lower bounds.

### Results

| Evaluation | Accepted long decisions | Correct | Long-override precision | Coverage / uncertainty |
| --- | ---: | ---: | ---: | --- |
| Five-fold grouped cross-validation | 29 | 23 | **79.31%** | Exact lower bound ≈63.2% |
| Policy-development split | 8 of 2,000 | 6 | 75.00% | 0.40% coverage |
| Sealed replacement holdout | **8 of 1,000** | **8** | **100.00%** | **0.80% coverage; exact lower bound 68.77%** |

The appropriate headline is **79.31% long-override precision on 29 accepted
cross-validation decisions**, not “79% accuracy.” Most requests receive the
four-second fallback and are not represented by that precision figure.

Cross-validation fold results were 4/6, 5/6, 3/4, 6/8, and 5/5. The strict
original 98% long-precision target was not met during selection.

### Failed and omitted gates

The frozen pragmatic release gate required at least 75% sealed precision, at
least 20 accepted sealed long examples, and no learned short decisions. The
candidate passed point precision and produced no learned short decisions, but
failed support with only 8 accepted examples.

Semantic inference measured 32.825 ms p95 on the benchmark system against the
30 ms target. Core Go tests, race detection, vet, loader/profile/cue fuzzing,
and cross-platform builds passed.

FETV evaluation and the independent audit were not run after the terminal
support failure. No result is claimed for either omitted evaluation.

The complete frozen evidence is available in
[LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json](evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json)
and the [evaluation status](evaluation/README.md).

## Technical specifications

The embedded runtime is pure Go with no ONNX Runtime, Python, CGO, external
model file, or inference-time network access. Immutable model data and pooled
independent workspaces support concurrent use.

The artifact loader supports formats v3, v4, and v5 and rejects malformed
dimensions, non-finite values, unknown labels, ambiguous legacy-v2 label
ordering, and artifacts larger than 64 MiB. The exact bytes loaded determine
the reported artifact SHA-256.

## Ethical and operational considerations

The model can increase generation cost but cannot independently reduce a
learned duration below the four-second fallback. This lowers under-duration
risk from learned decisions at the expense of occasionally unnecessary
extensions. Explicit user settings remain authoritative when valid.

Prompt content is processed locally at inference time. Applications remain
responsible for prompt handling, logging, access control, safety checks, price
updates, and provider compliance.

## License, attribution, and further information

The model is CC-BY-4.0 with VideoUFO attribution. The compact teacher encoder
is Apache-2.0. DiffusionDB is CC0; FETV is evaluation-only and CC-BY-4.0.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) or run
`steady-picker licenses`.

Further information:

- [README](README.md)
- [Semantic dataset card](docs/SEMANTIC_DATASET_CARD.md)
- [Methodology](docs/METHODOLOGY.md)
- [Artifact format](docs/ARTIFACT_V5.md)
- [v0.2.0 release](https://github.com/hensmth/steady-picker/releases/tag/v0.2.0)
