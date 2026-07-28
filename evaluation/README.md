# settings-v2 evaluation status

`settings-v2` is not releasable. The fixed 32-candidate, five-fold grouped
cross-validation grid completed against training-code commit
`c35c3fae51816d5aca5a8b4d68e6df4f823eb20f`. No candidate met the joint
short/long precision constraint:

- candidates evaluated: 32
- grouped folds per candidate: 5
- eligible candidates: 0
- long overrides accepted across all 160 fold evaluations: 0
- best mean short precision: 0.9753768883
- best mean cost-weighted selective utility: 0.1993333333

The complete machine-readable result is `cross-validation.json` (SHA-256
`3e4a2abff20bc66b64c89d35f105dd8e4bdde4f631e9aebfd6d1669d33d4488f`).
It was produced with Go 1.26.0. Teacher labels were produced with Hermes Agent
0.19.0 through the manifest-recorded provider and model.

Because hyperparameter selection failed, no candidate was fitted on the final
splits, no locked-test, FETV, or independent-audit result is claimed, and
`RELEASE_GATE_PASSED` does not exist. A provisional holdout was inspected during
engineering diagnosis after this failure and is therefore invalid for any
future release. A later attempt must use a genuinely untouched corpus and
pre-register any revised model family before evaluation.

Reproduction command:

```sh
python3 scripts/cross_validate.py \
  --binary ./steady-picker \
  --folds-dir ./folds \
  --work-dir ./grid-work \
  --source-manifest-sha256 9f493c245da7a6427ac5886ca87172b40ca83fa5d52ce3ad9572200c4ef267f3 \
  --training-code-commit c35c3fae51816d5aca5a8b4d68e6df4f823eb20f \
  --output ./cross-validation.json
```
