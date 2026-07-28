# Model Card: settings-v2 pragmatic candidate

## Model details

The v5 `settings-v2` candidate is a purpose-built, quantized semantic encoder:
two Transformer layers, hidden size 128, four attention heads, FFN size 512,
an 8,192-token WordPiece vocabulary, and a deterministic 96-token maximum.
Separate binary heads predict `short` and `long`; structured auxiliary heads
represent action beats, ordered progression, transformation, and dependent
actions. Inference is pure Go, offline, and self-contained.

The artifact carries the `long-only-pragmatic-v2` decision-policy marker.
Runtime policy suppresses every learned short decision. Explicit duration
requests remain deterministic, and `medium` remains the four-second fallback.

## Intended use

This is a conservative duration selector for quota-aware video generation. It
may extend a request from four to six seconds when the frozen long head is
accepted. It cannot shorten a request using a learned decision. Aspect ratio
and resolution are deterministic and outside the model.

It is not a safety classifier, content moderator, general semantic encoder, or
guarantee of the objectively correct duration.

## Training data and labels

The governed corpus contains 15,000 prompts: 14,000 development rows and a
separate 1,000-row sealed replacement holdout. Rights-cleared VideoUFO
CC-BY-4.0 captions and DiffusionDB CC0 prompts are filtered for identities,
URLs, explicit technical settings, unsafe content, exact duplicates, and
near-duplicate clusters.

Three blind, shuffled Sol Ultra teacher passes return only a duration label and
structured temporal fields. Only unanimous labels train or calibrate an
override; disagreement expects fallback. Provider, model, effort, source
revisions, votes, hashes, exclusions, and split membership are recorded.

## Selection and frozen candidate

Five-fold grouped cross-validation used only the 14,000 development rows. The
pragmatic candidate was fixed in advance as grid candidate 8:

- learning rate: 0.0007
- long-class weight: 0.75
- contrastive-loss weight: 0.25
- focal-loss strength: 1.0

The exact rerun accepted 29 learned-long examples and correctly labelled 23:
79.31% pooled precision. Fold results were 4/6, 5/6, 3/4, 6/8, and 5/5.
The one-sided exact 95% lower confidence bound is approximately 63.2%.

This does **not** satisfy the original 98% strict gate. That failure remains
recorded separately. The pragmatic policy instead requires at least 75%
locked-test precision and at least 20 accepted long examples, with no learned
short decisions. Its final long threshold maximizes coverage on the dedicated
policy-development split subject to the same 75% precision target. The sealed
holdout, FETV evaluation, and independent audit
remain release-blocking; no result is claimed until their checked-in gate
marker exists.

## Cost and limitations

At current profile estimates, each accepted extension adds $0.10 at 480p or
$0.14 at 720p compared with the four-second fallback. The CV result implies
roughly one unnecessary extension in five accepted decisions; that is the
explicit tradeoff of the pragmatic policy.

Results may degrade on multilingual, unusual, adversarial, or shifted input.
Cross-validation is not a production accuracy guarantee. No conformal
guarantee is claimed under distribution shift.

## License and attribution

The model is CC-BY-4.0 with VideoUFO attribution. The compact teacher encoder
is Apache-2.0. DiffusionDB is CC0; FETV is evaluation-only and CC-BY-4.0.
See `THIRD_PARTY_NOTICES.md`.
