# SteadyPicker v0.2.0

This release remains a draft until all locked model, engineering, integration,
attestation, and live-generation gates pass.

Highlights:

- strict provenance-carrying artifact format v3;
- immutable ordinary Go memory and concurrent pooled workspaces;
- configurable provider capability and budget profiles;
- quota-safe field-specific precedence and conflict/negation parsing;
- embedded `settings-v2`, profile, attribution, and health tooling;
- deterministic rights-cleared corpus, training, calibration, and evaluation;
- SHA-pinned cross-platform CI, SBOMs, and build attestations.

Migration: v2 artifacts from SteadyPicker v0.1 have ambiguous label order and
are intentionally rejected. Keep v0.1 for rollback or retrain to artifact v3.
