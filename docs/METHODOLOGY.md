# settings-v2 methodology

## Capability versus budget

The model predicts semantic duration only. Profiles translate semantics into
application settings. This avoids encoding a provider's current API limits into
learned weights.

The quota-safe profile is intentionally narrower than xAI's video API:
xAI documents 1–15 second generation and its current 480p/720p per-second
pricing, while this application profile permits only 2/4/6 seconds and caps at
six. Pricing is dated in every result and should be updated as metadata rather
than silently changing historical artifacts.

- https://docs.x.ai/developers/model-capabilities/video/generation
- https://docs.x.ai/developers/pricing

## Data governance

Only VideoUFO (CC-BY-4.0) and DiffusionDB (CC0) enter training. FETV
(CC-BY-4.0) is external evaluation only. VidProM, Panda-70M, OpenVid, WebVid,
and other non-commercial or upstream-ambiguous sources are excluded.

The source manifest pins repository revisions, raw/corpus digests, selection
seed, filters, clustering rule, row counts, and split membership. Teacher
manifests record exact Hermes, provider, and model values. OpenAI's Services
Agreement assigns customers responsibility for input rights and ownership of
outputs; it does not replace source-license review.

The semantic retraining corpus has 15,000 prompts: 4,000 governed non-locked
rows from the earlier corpus, 10,000 fresh duplicate-excluded rows, and a
separate 1,000-row replacement holdout. The fresh rows contain 4,000 broad,
4,000 high-temporal-complexity, and 2,000 temporal-cue hard-negative prompts.
The deterministic pre-annotation score never assigns a semantic label. Every
prior corpus, pilot, audit, and inspected test prompt is an exclusion source.

Three blind Sol Ultra passes produce only a duration label, action-beat count,
and ordered-progression, transformation, and dependent-action flags. Label
presentation and row order are independently permuted. A duration label enters
training only on three-of-three agreement; disagreement is represented as
ambiguous and expects the safe medium fallback. Usage evidence, checkpoints,
provider, model, effort, revisions, hashes, exclusions, and membership are
recorded without retaining free-form reasoning.

- https://huggingface.co/datasets/WenhaoWang/VideoUFO
- https://github.com/poloclub/diffusiondb
- https://github.com/llyx97/FETV
- https://openai.com/policies/services-agreement/

Dataset documentation follows the structure proposed by *Datasheets for
Datasets*; model documentation follows *Model Cards for Model Reporting*.

- https://arxiv.org/abs/1803.09010
- https://arxiv.org/abs/1810.03993

## Selection and calibration

The 14,000 non-locked rows are cluster-grouped into 10,000 training, 1,000
probability-calibration, 1,000 conformal-calibration, and 2,000
policy-development rows. A sealed validator reveals only whether the replacement
holdout has sufficient unanimous short/long support.

When the initial support check failed, one new holdout was selected
prospectively without teacher labels: 400 simple/single-beat candidates, 100
bridge cases, and 500 explicit progression/transformation candidates, balanced
as closely as the fully excluded source pools allowed. The failed holdout became
an exclusion source. Because this replacement is support-enriched, aggregate
locked metrics describe that designed mixture, not a natural deployment
prevalence; reporting must retain per-stratum results.

An Apache-2.0 `paraphrase-MiniLM-L3-v2` encoder is adapted with SetFit-style
contrastive pairs and used only as a teacher. The shipped student is a
purpose-built two-layer Transformer with hidden size 128, four attention heads,
FFN size 512, an 8,192-token WordPiece vocabulary, a 96-token maximum, separate
short/long heads, and temporal auxiliary heads. Per-output-channel INT8 weights
and float32 layer normalization keep inference self-contained in pure Go.

All hyperparameter selection occurs in five grouped folds inside the training
split. The fixed 16-candidate grid covers learning rate, long-class weight,
contrastive-loss weight, and focal-loss strength. Candidate ordering is fixed;
the short head uses its fold-specific inverse-frequency weight and the long
grid values multiply its independently derived fold-specific class-balance
weight. The best cost-weighted selective utility must first satisfy pooled
short/long precision, minimum accepted counts, and per-fold support
constraints. Ties resolve by lower fold variance and then candidate order.

After selection, temperature scaling uses a dedicated probability-calibration
split. Class-specific conformal quantiles use a separate calibration split.
Short and long probability thresholds are searched on policy-development data.
Only then is the locked test opened.

Class-conditional conformal sets support conservative in-domain acceptance, not
a distribution-free claim after deployment shift. External FETV metrics are
reported empirically and never used for tuning. This reporting distinction is
consistent with adaptive conformal classification and risk-controlling
prediction-set literature.

- https://arxiv.org/abs/2006.02544
- https://arxiv.org/abs/2101.02703

## Engineering and supply chain

The artifact parser validates all lengths and metadata before allocation and
rejects artifacts over 64 MiB. Format v5 fixes the complete semantic
architecture, tokenizer, quantization, provenance, and calibration layout while
retaining v3/v4 loading. Inference uses immutable ordinary Go slices and
independent pooled workspaces, with no ONNX runtime, Python, CGO, network access,
or external model file. The release target is zero steady-state allocations,
at most 30 milliseconds p95 on the deployment-class CPU, and an
executable/model below 32 MiB.

Race and fuzz testing follow the Go project's guidance:

- https://go.dev/doc/articles/race_detector
- https://go.dev/doc/tutorial/fuzz

GitHub Actions are pinned to full commit SHAs. Draft releases carry checksums,
SPDX SBOMs, build/model attestations, and the exact evaluation evidence.
Publication remains manual.

- https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
- https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
