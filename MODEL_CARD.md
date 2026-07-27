# SteadyPicker settings-v1

## Purpose

`settings-v1.bin` is a local, network-free duration classifier for public video
generation prompts. It is not a general language model. The surrounding
`balanced-v1` policy owns resolution, aspect ratio, quota caps, and fallback
behavior.

## Training data

- 750 deduplicated prompts sampled from DiffusionDB's CC0 prompt metadata.
- Duration labels (`d2`, `d4`, `d6`) produced offline by Hermes 0.19.0 using
  OpenAI Codex.
- No production prompts, account data, generated media, or credentials.

The trained model is committed under `models/` and embedded in released
binaries. The source corpus and teacher responses are excluded from Git. The
model contains hashed byte n-gram embeddings and cannot reproduce training
prompts.

## Validation

A deterministic, stratified five-fold evaluation predicted every row with a
model that had not trained on that row:

- 750 total held-out predictions.
- 544 top-1 correct (72.5%).
- 25 policy-accepted overrides (3.3% coverage).
- 25/25 accepted overrides matched the teacher (100% selective accuracy).

Top-1 accuracy is not sufficient for unrestricted use. Production acceptance
therefore requires a singleton conformal set, calibrated confidence of at least
0.80, and a matching short-beat or long-sequence cue. Otherwise the engine
returns the fixed 4-second fallback.

On a separate balanced 180-row FETV sample, the final model made no accepted
override. This is conservative domain-shift behavior, not evidence of broad
FETV accuracy.

## Limits

- English cue coverage is intentionally narrow.
- A prompt-injection phrase cannot alter policy, but explicit supported
  settings in prompt text are honored.
- The 100% selective result is based on only 25 accepted cross-validation
  predictions and should be monitored as more evaluation data is collected.
- Never call the raw classifier without the `balanced-v1` policy gate.

## Artifact

Version: `v1`

Artifact: `models/settings-v1.bin`

Expected SHA-256:

```text
0358b5262d27288b7c05cbfb40ba9de6d8771f0d7f0ac085e27e3561672c9b4d
```
