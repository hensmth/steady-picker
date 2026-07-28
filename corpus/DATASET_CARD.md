# Dataset Card: SteadyPicker settings-v2

## Dataset summary

This prompt-only dataset supports semantic video-duration classification. It
contains exactly 5,000 commercially reusable English prompts: 3,000 VideoUFO
brief captions and 2,000 DiffusionDB prompts.

## Collection and processing

The deterministic build records source revisions and SHA-256 digests. It
stratifies VideoUFO by topic and dynamic-degree band, restricts DiffusionDB to
low-NSFW rows, removes URLs/emails/handles and technical generation settings,
normalizes Unicode/whitespace, removes exact duplicates, and groups
near-duplicates using normalized token 3-gram Jaccard similarity `>=0.85`.

To make the locked test capable of measuring selective long-duration decisions,
the build also includes a disclosed temporal-complexity stratum. This is a
deterministic, rubric-derived text score, not a teacher label: 550 VideoUFO rows
and 1,450 DiffusionDB rows are selected from candidates scoring at least three.
VideoUFO selection remains round-robin across topic and dynamic-degree strata;
the larger DiffusionDB share reflects the available candidate pools. The
remaining 3,000 rows are sampled without that requirement. Pilot annotations
used to validate corpus design are excluded from every released split.

## Annotation

Hermes uses the recorded OpenAI provider and model locally. Two blind passes
shuffle examples and label presentation independently. Disagreements receive a
third pass. The released rows contain votes and final labels, but no free-form
reasoning. Weighted Cohen's kappa must be at least 0.75 and unresolved rows at
most 2%.

## Splits

Clusters never cross splits. Rows are stratified by source and final label into
3,000 train, 300 probability calibration, 300 conformal calibration, 400 policy
development, and 1,000 locked test rows. Unresolved rows expect fallback and
never enter fitting or calibration.

## Uses and limitations

Use for conservative video-duration research and reproduction of
`settings-v2`. Do not treat labels as objective video quality judgments or use
the dataset for content safety decisions. Distribution shift and multilingual
prompts require separate evaluation.

## Licensing

The combined published corpus and labels are CC-BY-4.0. VideoUFO remains
CC-BY-4.0 and must be attributed. DiffusionDB is CC0. See
`THIRD_PARTY_NOTICES.md`.
