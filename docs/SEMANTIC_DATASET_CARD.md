# Dataset Card: settings-v2 Semantic Training Data

## Dataset summary

This governed prompt collection supports semantic video-duration selection for
SteadyPicker `settings-v2`. The workflow contains 15,000 English prompts:
14,000 development examples and a separate 1,000-example replacement holdout.

The dataset labels textual temporal complexity. It does not contain generated
videos and does not measure visual quality or human viewing preference.

## Sources and licensing

| Source | Use | License |
| --- | --- | --- |
| VideoUFO | Training, calibration, policy development, and holdout prompts | CC-BY-4.0 |
| DiffusionDB | Training, calibration, policy development, and holdout prompts | CC0 |
| FETV | External evaluation only; not used for training or tuning | CC-BY-4.0 |

Non-commercial or upstream-ambiguous sources are excluded. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for attribution.

## Composition

The 14,000 development prompts consist of:

- 4,000 governed non-locked examples retained from earlier data;
- 4,000 fresh broad prompts;
- 4,000 fresh high-temporal-complexity prompts; and
- 2,000 fresh temporal-cue hard negatives.

The remaining 1,000 prompts form a separately selected, support-enriched
replacement holdout. Aggregate holdout metrics therefore describe this
designed mixture rather than natural deployment prevalence.

Development rows are grouped by normalized token 3-gram near-duplicate
clusters and assigned to:

| Split | Rows |
| --- | ---: |
| Training | 10,000 |
| Probability calibration | 1,000 |
| Conformal calibration | 1,000 |
| Policy development | 2,000 |
| Replacement holdout | 1,000 |

Clusters do not cross splits. Exact and near duplicates from prior corpora,
pilots, audits, inspected tests, and retired holdouts are excluded.

## Collection and filtering

Source revisions and content hashes are pinned by the training manifests.
Collection removes or filters URLs, email addresses, handles, explicit
technical generation settings, unsafe content, malformed text, exact
duplicates, and near-duplicate clusters.

Pre-annotation temporal-complexity strata support broad, progression-heavy,
and hard-negative sampling. These deterministic strata do not assign semantic
labels.

## Annotation

Three blind Sol Ultra teacher passes independently process shuffled rows with
permuted label presentation. Each pass returns only:

- duration label;
- action-beat count;
- ordered-progression flag;
- transformation flag; and
- dependent-action flag.

A learned duration label requires three-of-three agreement. Every disagreement
is marked ambiguous and expects the safe `medium` fallback. These are
AI-adjudicated labels, not human annotations or rendered-video judgements.

## Intended uses

The dataset supports reproduction and evaluation of conservative semantic
duration selection. It may also support research into selective
classification, temporal-language cues, and quota-aware fallback policies
under the source licenses.

It should not be used as:

- objective ground truth for video quality or ideal duration;
- a content-safety or moderation dataset;
- evidence of human preference;
- a representative sample of production prevalence; or
- a multilingual benchmark without separate validation.

## Risks and limitations

Teacher-model preferences and systematic errors may be inherited by the
labels. Unanimity reduces ambiguity but does not establish correctness.
Sampling intentionally enriches temporal complexity, and the replacement
holdout is support-enriched. Results may not transfer to unusual, adversarial,
multilingual, or distribution-shifted prompts.

The public repository's `corpus/` directory documents an earlier 5,000-prompt
sparse-model corpus. It is not the semantic model's 15,000-prompt dataset.
The semantic workflow and frozen release evidence are documented here, in the
[methodology](METHODOLOGY.md), and in the
[released diagnostics](../evaluation/LONG_ONLY_PRAGMATIC_V2_DIAGNOSTICS.json).
