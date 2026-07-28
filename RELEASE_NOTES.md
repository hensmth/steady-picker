# SteadyPicker v0.2.0

SteadyPicker v0.2.0 ships the self-contained, offline long-only semantic model.

The model:

- can extend a request from four to six seconds;
- never makes a learned two-second decision;
- honors explicit two-second requests; and
- keeps deterministic aspect-ratio, resolution, and safe-fallback behavior.

Observed frozen evaluation:

- grouped CV: 23/29 accepted long decisions correct (79.31%);
- sealed test: 8/8 accepted long decisions correct (100%);
- sealed learned-override coverage: 0.8%;
- development-host latency: 32.825 ms p95.

Assets include Linux amd64/arm64, macOS amd64/arm64, and Windows amd64
binaries, the standalone model, SHA-256 checksums, SPDX SBOMs, build
attestations, the model card, notices, and evaluation diagnostics.
