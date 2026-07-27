# SteadyPicker

SteadyPicker is a tiny, deterministic video-settings engine. It converts a
generation prompt into quota-aware duration, aspect-ratio, and resolution
settings without making a production network or LLM call.

The released CLI includes the trained `settings-v1` model, so one executable is
all you need.

## Quick start

Install with Go:

```bash
go install github.com/hensmth/steady-picker/cmd/steady-picker@latest
```

Or download a ready-to-run binary from the
[latest release](https://github.com/hensmth/steady-picker/releases/latest) for
Linux, macOS, or Windows.

Send one JSON object over stdin:

```bash
printf '%s\n' \
  '{"prompt":"a flower transforming in a timelapse","mode":"text-to-video"}' |
  steady-picker predict
```

Output:

```json
{"duration":6,"aspect_ratio":"16:9","resolution":"480p","source":"model","confidence":0.9999,"model_version":"v1","policy_version":"balanced-v1"}
```

Prompts are read from stdin, not command-line arguments. Prediction is local,
network-free, and normally completes in milliseconds.

## Policy

- Safe fallback: 4 seconds, 16:9, 480p.
- Maximum duration: 6 seconds.
- Default resolution: 480p.
- `720p` requires `720p`, `HD output`, or `render in HD` in the prompt.
- Decorative wording such as `4K style`, `cinematic`, or `high quality` stays
  at 480p.
- Image-to-video requests retain the source frame with `auto`.
- Learned 2- or 6-second overrides require a singleton conformal set, at least
  0.80 calibrated confidence, and a matching semantic cue.

Ambiguous or out-of-distribution prompts keep the conservative fallback.

## Input

`predict` accepts:

```json
{
  "prompt": "a quick vertical video of a bird taking flight",
  "mode": "text-to-video",
  "image_aspect_ratio": 1.777
}
```

- `prompt` is required.
- `mode` is `text-to-video` or `image-to-video`.
- `image_aspect_ratio` is optional and only relevant to image-to-video.
- Input is limited to one JSON line of at most 16 KiB.

Use a custom model when needed:

```bash
steady-picker predict --model ./custom-model.bin --model-version custom-v1
```

## Go library

```go
package main

import (
	"fmt"
	"log"

	steady "github.com/hensmth/steady-picker"
)

func main() {
	model, err := steady.LoadDefault()
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	result := steady.PickSettings(model, steady.PickRequest{
		Prompt: "a quick wink in a portrait video",
		Mode:   "text-to-video",
	}, steady.DefaultModelVersion)
	fmt.Printf("%+v\n", result)
}
```

`Load(path)` memory-maps an external model. `LoadBytes(data)` supports models
embedded by another application.

## Training

Training data uses one label per line:

```text
__label__d2 A quick shot of a bird taking flight
__label__d4 A dancer turns toward the camera
__label__d6 A timelapse showing a flower fully bloom
```

```bash
steady-picker train --input training.txt --output custom-model.bin
steady-picker evaluate --model custom-model.bin --test held-out.txt
```

Training is seeded and single-worker by default, so identical inputs and
parameters produce identical model checksums. The scripts directory contains
optional public-prompt acquisition, Hermes teacher labelling, deterministic
splitting, and cross-validation helpers. Teacher labelling runs locally by
default and accepts `--ssh-host` for a remote Hermes installation.

## Development

```bash
go test ./...
go test -race ./...
go vet ./...
```

Released binaries are built for:

- Linux: amd64 and arm64
- macOS: amd64 and arm64
- Windows: amd64

See [MODEL_CARD.md](MODEL_CARD.md) for model provenance, evaluation, intended
use, and limitations.

## Origin and license

The classifier is derived from
[`xDarkicex/steady`](https://github.com/xDarkicex/steady) and retains its MIT
license. SteadyPicker adds independent calibration data, complete one-vs-all
updates, corrected averaged-embedding gradients, deterministic training,
strict artifact validation, and the quota policy.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
