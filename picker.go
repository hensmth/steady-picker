package steady

import (
	"regexp"
	"strconv"
	"strings"
)

const (
	PolicyVersion     = "balanced-v1"
	DefaultResolution = "480p"
	MaximumDuration   = 6
	FallbackDuration  = 4
	FallbackAspect    = "16:9"
	MinimumConfidence = 0.80
)

var PresetLabels = []string{"d2", "d4", "d6"}

type PickRequest struct {
	Prompt           string  `json:"prompt"`
	Mode             string  `json:"mode"`
	ImageAspectRatio float64 `json:"image_aspect_ratio,omitempty"`
}

type PickResult struct {
	Duration      int     `json:"duration"`
	AspectRatio   string  `json:"aspect_ratio"`
	Resolution    string  `json:"resolution"`
	Source        string  `json:"source"`
	Confidence    float32 `json:"confidence"`
	ModelVersion  string  `json:"model_version"`
	PolicyVersion string  `json:"policy_version"`
}

var (
	durationPattern  = regexp.MustCompile(`(?i)\b([1-9]|1[0-5])\s*(?:s|sec|second)s?\b`)
	hdPattern        = regexp.MustCompile(`(?i)(?:\b720p\b|\bHD\s+output\b|\brender\s+in\s+HD\b)`)
	portraitPattern  = regexp.MustCompile(`(?i)(?:\b9\s*:\s*16\b|\bvertical\s+(?:video|output|frame)\b|\bportrait\s+(?:video|output|frame)\b)`)
	squarePattern    = regexp.MustCompile(`(?i)(?:\b1\s*:\s*1\b|\bsquare\s+(?:video|output|frame)\b)`)
	landscapePattern = regexp.MustCompile(`(?i)(?:\b16\s*:\s*9\b|\blandscape\s+(?:video|output|frame)\b|\bwidescreen\s+(?:video|output|frame)\b)`)
	shortBeatPattern = regexp.MustCompile(`(?i)\b(?:quick|brief|blink|wink|flash|burst)\b`)
	longBeatPattern  = regexp.MustCompile(`(?i)\b(?:then|timelapse|time-lapse|transformation|transforming|morph|before and after|sequence|progression)\b`)
)

func PickSettings(model *Model, request PickRequest, modelVersion string) PickResult {
	result := PickResult{
		Duration: FallbackDuration, AspectRatio: FallbackAspect,
		Resolution: DefaultResolution, Source: "fallback",
		ModelVersion: modelVersion, PolicyVersion: PolicyVersion,
	}
	prompt := strings.TrimSpace(request.Prompt)
	if hdPattern.MatchString(prompt) {
		result.Resolution = "720p"
	}

	explicitDuration := 0
	if match := durationPattern.FindStringSubmatch(prompt); len(match) == 2 {
		explicitDuration, _ = strconv.Atoi(match[1])
		if explicitDuration > MaximumDuration {
			explicitDuration = MaximumDuration
		}
	}
	explicitAspect := explicitAspect(prompt)

	if request.Mode == "image-to-video" && request.ImageAspectRatio > 0 {
		explicitAspect = "auto"
	}

	if model != nil {
		model.SetLabelNames(PresetLabels)
		debug := model.ClassifyDebug(prompt)
		bestIndex, bestConfidence := bestPrediction(debug.Calibrated)
		if !debug.IsEmpty && len(debug.Kinds) == 1 &&
			bestIndex >= 0 && bestConfidence >= MinimumConfidence {
			duration, ok := decodeDuration(PresetLabels[bestIndex])
			if ok && hasDurationEvidence(prompt, duration) {
				result.Duration = duration
				result.Confidence = bestConfidence
				result.Source = "model"
			}
		}
	}
	if explicitDuration > 0 {
		result.Duration = explicitDuration
		result.Source = "explicit"
	}
	if explicitAspect != "" {
		result.AspectRatio = explicitAspect
	}
	if result.Duration > MaximumDuration {
		result.Duration = MaximumDuration
	}
	return result
}

func hasDurationEvidence(prompt string, duration int) bool {
	switch duration {
	case 2:
		return shortBeatPattern.MatchString(prompt)
	case 4:
		return false // Four seconds is already the conservative fallback.
	case 6:
		return longBeatPattern.MatchString(prompt)
	default:
		return false
	}
}

func explicitAspect(prompt string) string {
	switch {
	case portraitPattern.MatchString(prompt):
		return "9:16"
	case squarePattern.MatchString(prompt):
		return "1:1"
	case landscapePattern.MatchString(prompt):
		return "16:9"
	default:
		return ""
	}
}

func decodeDuration(label string) (int, bool) {
	duration, err := strconv.Atoi(strings.TrimPrefix(label, "d"))
	if err != nil {
		return 0, false
	}
	return duration, duration == 2 || duration == 4 || duration == 6
}

func bestPrediction(probabilities []float32) (int, float32) {
	bestIndex := -1
	var best float32
	for index, probability := range probabilities {
		if probability > best {
			bestIndex, best = index, probability
		}
	}
	return bestIndex, best
}
