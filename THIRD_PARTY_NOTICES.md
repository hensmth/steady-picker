# Third-party notices

## steady

SteadyPicker derives its original classifier structure from
[`xDarkicex/steady`](https://github.com/xDarkicex/steady), Copyright its
contributors, licensed under MIT. The MIT terms are retained in `LICENSE`.

## VideoUFO

The settings-v2 corpus includes 3,000 sanitized brief captions from VideoUFO by
Wenhao Wang and contributors. VideoUFO is licensed under CC-BY-4.0.

- Dataset: https://huggingface.co/datasets/WenhaoWang/VideoUFO
- Paper: https://arxiv.org/abs/2503.01739
- Modifications: prompt-only selection, normalization, filtering,
  deduplication, near-duplicate clustering, teacher labels, and split metadata.

## DiffusionDB

The settings-v2 corpus includes 2,000 sanitized prompts from DiffusionDB by the
Stanford HAI Visualization Group. DiffusionDB is dedicated to the public domain
under CC0-1.0.

- Project: https://github.com/poloclub/diffusiondb

## FETV

FETV by Yuanxin Liu and contributors is used only for external evaluation and
is licensed under CC-BY-4.0.

- Project: https://github.com/llyx97/FETV
- Paper: https://arxiv.org/abs/2311.01813

## paraphrase-MiniLM-L3-v2

The semantic retraining pipeline uses
[`sentence-transformers/paraphrase-MiniLM-L3-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2)
at its recorded immutable revision as the teacher encoder. The teacher is
licensed under Apache-2.0. It is used during training only; its weights are not
shipped in the SteadyPicker executable or student artifact.

No VidProM, Panda-70M, OpenVid, WebVid, or other non-commercial or
upstream-ambiguous dataset is included.
