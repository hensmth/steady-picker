# settings-v2 evaluation status

The original strict v0.2 selection remains failed and unreleased. Its evidence
is preserved in this directory and must not be reinterpreted as a pass.

## Pragmatic long-only candidate

Grid candidate 8 was frozen for the separate
`long-only-pragmatic-v2` policy:

| Fold | Correct | Accepted | Precision |
| --- | ---: | ---: | ---: |
| 1 | 4 | 6 | 66.67% |
| 2 | 5 | 6 | 83.33% |
| 3 | 3 | 4 | 75.00% |
| 4 | 6 | 8 | 75.00% |
| 5 | 5 | 5 | 100.00% |
| **Pooled** | **23** | **29** | **79.31%** |

The exact clean rerun matches the frozen evidence. The one-sided exact 95%
lower confidence bound is approximately 63.2%. Learned short decisions are
disabled at runtime; an explicit two-second request is still honored.

At quota-safe profile estimates, a learned six-second extension adds $0.10 at
480p or $0.14 at 720p over fallback. With six CV errors, the observed wasted
incremental spend was $0.60 at 480p-equivalent pricing.

## Release gate

The pragmatic release requires:

- locked long precision at least 75%;
- at least 20 accepted locked long examples;
- zero learned short decisions on locked and FETV data;
- at least 75% agreement on the audit's accepted-long slice; and
- zero severe short-under-duration audit errors.

The sealed test is opened only after source and artifact are frozen. Until
`LONG_ONLY_PRAGMATIC_V2_PASSED` exists, the candidate must not be tagged,
released, embedded, or promoted. Cross-validation evidence alone is not a
release result.
