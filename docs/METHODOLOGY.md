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

The corpus includes a pre-annotation temporal-complexity stratum so the locked
test has enough progression and dependent-action examples to measure selective
long decisions. The deterministic score counts rubric-derived temporal and
action cues; it never assigns a semantic label. Candidate-pool sizes, the
source-specific quotas, and the scoring definition are published. Pilot
annotations used to validate prevalence are permanently excluded from model
fitting, calibration, policy development, and locked evaluation.

- https://huggingface.co/datasets/WenhaoWang/VideoUFO
- https://github.com/poloclub/diffusiondb
- https://github.com/llyx97/FETV
- https://openai.com/policies/services-agreement/

Dataset documentation follows the structure proposed by *Datasheets for
Datasets*; model documentation follows *Model Cards for Model Reporting*.

- https://arxiv.org/abs/1803.09010
- https://arxiv.org/abs/1810.03993

## Selection and calibration

All hyperparameter selection occurs in five grouped folds inside the training
split. Candidate ordering is fixed; the best mean cost-weighted selective
utility must first satisfy short/long precision constraints. Ties resolve by
lower fold variance, then smaller artifact.

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
rejects artifacts over 64 MiB. Inference uses immutable ordinary Go slices and
independent pooled workspaces. The release target is zero steady-state
allocations, under 50 microseconds, and an embedded artifact below 10 MiB.

Race and fuzz testing follow the Go project's guidance:

- https://go.dev/doc/articles/race_detector
- https://go.dev/doc/tutorial/fuzz

GitHub Actions are pinned to full commit SHAs. Draft releases carry checksums,
SPDX SBOMs, and build/model attestations. Publication remains manual and
blocked unless the locked release marker exists.

- https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
- https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
