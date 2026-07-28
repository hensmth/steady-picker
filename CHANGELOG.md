# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Quantized v5 semantic artifacts with a two-layer Transformer, structured
  temporal heads, and v3/v4 loading compatibility.
- A separately identified `long-only-pragmatic-v2` decision policy and release
  gate.
- Strict provenance-carrying v3 artifacts and embedded `settings-v2`.
- Configurable capability/budget profiles and `quota-safe-v2`.
- Field-specific decision sources, stable reasons, cost estimates, and dated
  pricing metadata.
- Rights-cleared corpus, multi-pass teacher, calibration, locked evaluation,
  and independent audit tooling.
- `inspect-model`, `health`, and `licenses` commands.

### Changed

- Learned short decisions are suppressed by the pragmatic policy; explicit
  two-second requests remain available.
- Replaced unsafe mapping and custom allocation with ordinary immutable Go
  slices and independent pooled workspaces.
- Learned labels are provider-neutral `short`, `medium`, and `long`.
- `predict` consumes newline-delimited JSON exclusively from stdin.

### Removed

- Mutable label state, lifecycle-dependent `Close`, shared scratch memory,
unsafe code, mmap, and third-party runtime dependencies.

## [0.2.0] — 2026-07-28

### Added

- Embedded long-only semantic model.
- Cross-platform release binaries, standalone model, checksums, SBOMs, and
  attestations.

### Safety

- Learned short decisions are disabled.
- Evaluation coverage and latency measurements are disclosed with the release.

## [0.1.0] — 2026-07-28

### Added

- Quota-aware duration, aspect-ratio, and resolution policy.
- Conservative model gate with fixed 4-second fallback.
- JSON-lines prediction and evaluation commands.
- Deterministic single-worker training and atomic model artifacts.
- Independent stratified calibration split and complete one-vs-all updates.
- Prompt acquisition, teacher labelling, balanced splits, and cross-validation
  scripts.
- Embedded `settings-v1` model with optional external-model overrides.
- Static release binaries for Linux, macOS, and Windows.

[0.1.0]: https://github.com/hensmth/steady-picker/releases/tag/v0.1.0
