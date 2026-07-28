package steady

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	ProfileQuotaSafeV2 = "quota-safe-v2"
	maxPromptBytes     = 16 << 10
)

type Mode string

const (
	TextToVideo  Mode = "text-to-video"
	ImageToVideo Mode = "image-to-video"
)

type PriceEstimate struct {
	Resolution   string `json:"resolution"`
	MicroUSDPerS int64  `json:"microusd_per_second"`
}

type ProfileConfig struct {
	Name               string            `json:"name"`
	Version            string            `json:"version"`
	AllowedDurations   []int             `json:"allowed_durations"`
	SemanticDurations  map[string]int    `json:"semantic_durations"`
	AllowedResolutions []string          `json:"allowed_resolutions"`
	AllowedAspects     []string          `json:"allowed_aspect_ratios"`
	DefaultDuration    int               `json:"default_duration"`
	MaximumDuration    int               `json:"maximum_duration"`
	DefaultResolution  string            `json:"default_resolution"`
	DefaultTextAspect  string            `json:"default_text_aspect_ratio"`
	PricingAsOf        string            `json:"pricing_as_of,omitempty"`
	Prices             []PriceEstimate   `json:"prices,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
}

type Profile struct {
	config ProfileConfig
	prices map[string]int64
}

func NewProfile(config ProfileConfig) (Profile, error) {
	config.AllowedDurations = slices.Clone(config.AllowedDurations)
	config.AllowedResolutions = slices.Clone(config.AllowedResolutions)
	config.AllowedAspects = slices.Clone(config.AllowedAspects)
	config.SemanticDurations = cloneMap(config.SemanticDurations)
	config.Metadata = cloneMap(config.Metadata)
	sort.Ints(config.AllowedDurations)
	if config.Name == "" || config.Version == "" || len(config.AllowedDurations) == 0 ||
		len(config.AllowedResolutions) == 0 || len(config.AllowedAspects) == 0 {
		return Profile{}, errors.New("steady: profile identity and capabilities are required")
	}
	if !strictPositiveUnique(config.AllowedDurations) ||
		!slices.Contains(config.AllowedDurations, config.DefaultDuration) ||
		!slices.Contains(config.AllowedDurations, config.MaximumDuration) ||
		config.MaximumDuration < config.DefaultDuration {
		return Profile{}, errors.New("steady: invalid duration capabilities")
	}
	if config.AllowedDurations[len(config.AllowedDurations)-1] > config.MaximumDuration {
		return Profile{}, errors.New("steady: allowed duration exceeds profile maximum")
	}
	if !uniqueValid(config.AllowedResolutions, validResolution) ||
		!uniqueValid(config.AllowedAspects, validAspect) ||
		!slices.Contains(config.AllowedResolutions, config.DefaultResolution) ||
		!slices.Contains(config.AllowedAspects, config.DefaultTextAspect) {
		return Profile{}, errors.New("steady: invalid resolution or aspect capabilities")
	}
	for _, label := range []string{"short", "medium", "long"} {
		duration, ok := config.SemanticDurations[label]
		if !ok || !slices.Contains(config.AllowedDurations, duration) || duration > config.MaximumDuration {
			return Profile{}, fmt.Errorf("steady: invalid %s semantic mapping", label)
		}
	}
	if len(config.SemanticDurations) != 3 {
		return Profile{}, errors.New("steady: semantic mapping must contain only short, medium, and long")
	}
	prices := make(map[string]int64, len(config.Prices))
	for _, price := range config.Prices {
		if !slices.Contains(config.AllowedResolutions, price.Resolution) || price.MicroUSDPerS < 0 {
			return Profile{}, errors.New("steady: invalid price estimate")
		}
		if _, exists := prices[price.Resolution]; exists {
			return Profile{}, errors.New("steady: duplicate price estimate")
		}
		prices[price.Resolution] = price.MicroUSDPerS
	}
	if len(prices) > 0 && config.PricingAsOf == "" {
		return Profile{}, errors.New("steady: dated pricing metadata is required")
	}
	return Profile{config: config, prices: prices}, nil
}

func QuotaSafeProfile() Profile {
	profile, err := NewProfile(ProfileConfig{
		Name: ProfileQuotaSafeV2, Version: "2",
		AllowedDurations:   []int{2, 4, 6},
		SemanticDurations:  map[string]int{"short": 2, "medium": 4, "long": 6},
		AllowedResolutions: []string{"480p", "720p"},
		AllowedAspects:     []string{"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"},
		DefaultDuration:    4, MaximumDuration: 6, DefaultResolution: "480p",
		DefaultTextAspect: "16:9", PricingAsOf: "2026-07-28",
		Prices: []PriceEstimate{{"480p", 50_000}, {"720p", 70_000}},
	})
	if err != nil {
		panic(err)
	}
	return profile
}

type PickRequest struct {
	Prompt                 string  `json:"prompt"`
	Mode                   Mode    `json:"mode"`
	Duration               int     `json:"duration,omitempty"`
	AspectRatio            string  `json:"aspect_ratio,omitempty"`
	Resolution             string  `json:"resolution,omitempty"`
	SourceMediaAspectRatio string  `json:"source_media_aspect_ratio,omitempty"`
	ImageAspectRatio       float64 `json:"image_aspect_ratio,omitempty"` // v0.1 compatibility
}

type PickResult struct {
	Duration              int     `json:"duration"`
	AspectRatio           string  `json:"aspect_ratio"`
	Resolution            string  `json:"resolution"`
	Source                string  `json:"source"`
	DurationSource        string  `json:"duration_source"`
	AspectRatioSource     string  `json:"aspect_ratio_source"`
	ResolutionSource      string  `json:"resolution_source"`
	Confidence            float32 `json:"confidence"`
	ModelVersion          string  `json:"model_version"`
	PolicyVersion         string  `json:"policy_version"`
	ProfileVersion        string  `json:"profile_version"`
	ArtifactSHA256        string  `json:"artifact_sha256"`
	Reasons               Reasons `json:"reasons"`
	EstimatedCostMicroUSD int64   `json:"estimated_cost_microusd"`
	PricingAsOf           string  `json:"pricing_as_of,omitempty"`
}

// Reasons is a small caller-owned value list with JSON-array encoding.
type Reasons struct {
	values [8]string
	count  uint8
}

func (r *Reasons) add(value string) {
	for index := uint8(0); index < r.count; index++ {
		if r.values[index] == value {
			return
		}
	}
	if r.count < uint8(len(r.values)) {
		r.values[r.count] = value
		r.count++
	}
}

// Strings returns a caller-owned slice.
func (r Reasons) Strings() []string {
	return append([]string(nil), r.values[:r.count]...)
}

func (r Reasons) MarshalJSON() ([]byte, error) {
	return json.Marshal(r.values[:r.count])
}

func (r *Reasons) UnmarshalJSON(data []byte) error {
	var values []string
	if err := json.Unmarshal(data, &values); err != nil {
		return err
	}
	if len(values) > len(r.values) {
		return errors.New("steady: too many reasons")
	}
	*r = Reasons{}
	for _, value := range values {
		if value == "" || len(value) > 128 {
			return errors.New("steady: invalid reason")
		}
		r.add(value)
	}
	return nil
}

var (
	durationPattern   = regexp.MustCompile(`(?i)\b([1-9]|1[0-5])\s*(?:s|sec|secs|second|seconds)\b`)
	aspectPattern     = regexp.MustCompile(`(?i)\b(1\s*:\s*1|16\s*:\s*9|9\s*:\s*16|4\s*:\s*3|3\s*:\s*4|3\s*:\s*2|2\s*:\s*3)\b`)
	resolutionPattern = regexp.MustCompile(`(?i)\b(480p|720p|1080p|4k)\b`)
	negationPattern   = regexp.MustCompile(`(?i)\b(?:not|no|without|don't|do not|isn't|is not|avoid)\b`)
)

func PickSettings(model *Model, profile Profile, request PickRequest) (PickResult, error) {
	if err := validateRequest(profile, request); err != nil {
		return PickResult{}, err
	}
	cfg := profile.config
	result := PickResult{
		Duration: cfg.DefaultDuration, AspectRatio: cfg.DefaultTextAspect,
		Resolution: cfg.DefaultResolution, DurationSource: "fallback",
		AspectRatioSource: "fallback", ResolutionSource: "fallback",
		ModelVersion: "none", PolicyVersion: cfg.Name, ProfileVersion: cfg.Version,
	}
	if model != nil {
		result.ModelVersion = model.metadata.ModelID
		result.ArtifactSHA256 = model.metadata.ArtifactSHA256
	}
	prompt := strings.TrimSpace(request.Prompt)

	if request.Duration > 0 {
		result.Duration = normalizeDuration(request.Duration, profile, &result.Reasons, "duration.request")
		result.DurationSource = "explicit"
	} else if values := affirmativeDurationCues(prompt); len(values) == 1 {
		result.Duration = normalizeDuration(values[0], profile, &result.Reasons, "duration.prompt")
		result.DurationSource = "explicit"
	} else if len(values) > 1 {
		result.Reasons.add("duration.conflict_fallback")
	} else if model != nil {
		if label, probability, accepted := model.pickDecision(prompt); accepted {
			result.Duration = cfg.SemanticDurations[label]
			result.DurationSource = "model"
			result.Confidence = probability
			if label == "short" {
				result.Reasons.add("duration.model_short")
			} else {
				result.Reasons.add("duration.model_long")
			}
		}
	}

	if request.AspectRatio != "" {
		result.AspectRatio = request.AspectRatio
		result.AspectRatioSource = "explicit"
		result.Reasons.add("aspect.request")
	} else if request.Mode == ImageToVideo && (request.SourceMediaAspectRatio != "" || request.ImageAspectRatio > 0) {
		result.AspectRatio = "auto"
		result.AspectRatioSource = "source_media"
		result.Reasons.add("aspect.preserve_source")
	} else if values := affirmativeCues(prompt, aspectPattern, normalizeAspect); len(values) == 1 &&
		slices.Contains(cfg.AllowedAspects, values[0]) {
		result.AspectRatio = values[0]
		result.AspectRatioSource = "explicit"
		result.Reasons.add("aspect.prompt")
	} else if len(values) > 0 {
		result.Reasons.add("aspect.conflict_or_unsupported_fallback")
	}

	if request.Resolution != "" {
		result.Resolution = strings.ToLower(request.Resolution)
		result.ResolutionSource = "explicit"
		result.Reasons.add("resolution.request")
	} else if values := affirmativeCues(prompt, resolutionPattern, strings.ToLower); len(values) == 1 &&
		slices.Contains(cfg.AllowedResolutions, values[0]) {
		result.Resolution = values[0]
		result.ResolutionSource = "explicit"
		result.Reasons.add("resolution.prompt")
	} else if len(values) > 0 {
		result.Reasons.add("resolution.conflict_or_unsupported_fallback")
	}

	result.Source = result.DurationSource
	if result.DurationSource == "fallback" {
		result.Reasons.add("duration.safe_fallback")
	}
	if result.AspectRatioSource == "fallback" {
		result.Reasons.add("aspect.safe_fallback")
	}
	if result.ResolutionSource == "fallback" {
		result.Reasons.add("resolution.safe_fallback")
	}
	result.PricingAsOf = cfg.PricingAsOf
	if price, ok := profile.prices[result.Resolution]; ok {
		result.EstimatedCostMicroUSD = int64(result.Duration) * price
	}
	return result, nil
}

func validateRequest(profile Profile, request PickRequest) error {
	if profile.config.Name == "" {
		return errors.New("steady: profile is not initialized")
	}
	if request.Mode != TextToVideo && request.Mode != ImageToVideo {
		return errors.New("steady: mode must be text-to-video or image-to-video")
	}
	if !utf8.ValidString(request.Prompt) || len(request.Prompt) > maxPromptBytes || strings.TrimSpace(request.Prompt) == "" {
		return errors.New("steady: prompt must be valid UTF-8, non-empty, and at most 16384 bytes")
	}
	if request.Duration < 0 || request.Duration > 1_000_000 {
		return errors.New("steady: invalid requested duration")
	}
	if request.AspectRatio != "" && !slices.Contains(profile.config.AllowedAspects, request.AspectRatio) {
		return errors.New("steady: requested aspect ratio is not supported by profile")
	}
	if request.Resolution != "" && !slices.Contains(profile.config.AllowedResolutions, strings.ToLower(request.Resolution)) {
		return errors.New("steady: requested resolution is not supported by profile")
	}
	if request.SourceMediaAspectRatio != "" && !validAspect(request.SourceMediaAspectRatio) {
		return errors.New("steady: invalid source media aspect ratio")
	}
	if math.IsNaN(request.ImageAspectRatio) || math.IsInf(request.ImageAspectRatio, 0) || request.ImageAspectRatio < 0 {
		return errors.New("steady: invalid image aspect ratio")
	}
	return nil
}

func normalizeDuration(value int, profile Profile, reasons *Reasons, prefix string) int {
	if value >= profile.config.MaximumDuration {
		if value > profile.config.MaximumDuration {
			reasons.add(prefix + "_clamped")
		}
		return profile.config.MaximumDuration
	}
	for _, allowed := range profile.config.AllowedDurations {
		if allowed >= value {
			if allowed != value {
				reasons.add(prefix + "_rounded_up")
			} else {
				reasons.add(prefix)
			}
			return allowed
		}
	}
	reasons.add(prefix + "_fallback")
	return profile.config.DefaultDuration
}

func affirmativeDurationCues(text string) []int {
	values := affirmativeCues(text, durationPattern, func(value string) string {
		fields := strings.Fields(value)
		return fields[0]
	})
	out := make([]int, 0, len(values))
	for _, value := range values {
		n, _ := strconv.Atoi(value)
		out = append(out, n)
	}
	return out
}

func affirmativeCues(text string, pattern *regexp.Regexp, normalize func(string) string) []string {
	matches := pattern.FindAllStringIndex(text, -1)
	seen := map[string]bool{}
	var values []string
	for _, match := range matches {
		start := max(0, match[0]-32)
		if negationPattern.MatchString(text[start:match[0]]) {
			continue
		}
		value := normalize(text[match[0]:match[1]])
		if !seen[value] {
			seen[value] = true
			values = append(values, value)
		}
	}
	return values
}

func normalizeAspect(value string) string {
	return strings.ReplaceAll(strings.ToLower(value), " ", "")
}
func validAspect(value string) bool {
	parts := strings.Split(value, ":")
	if len(parts) != 2 {
		return false
	}
	a, e1 := strconv.Atoi(parts[0])
	b, e2 := strconv.Atoi(parts[1])
	return e1 == nil && e2 == nil && a > 0 && b > 0 && a <= 100 && b <= 100
}
func validResolution(value string) bool {
	if !strings.HasSuffix(strings.ToLower(value), "p") {
		return false
	}
	n, err := strconv.Atoi(strings.TrimSuffix(strings.ToLower(value), "p"))
	return err == nil && n >= 144 && n <= 8640
}
func strictPositiveUnique(values []int) bool {
	for i, value := range values {
		if value <= 0 || (i > 0 && values[i-1] == value) {
			return false
		}
	}
	return true
}
func uniqueValid(values []string, valid func(string) bool) bool {
	seen := map[string]bool{}
	for _, value := range values {
		if seen[value] || !valid(value) {
			return false
		}
		seen[value] = true
	}
	return true
}
func cloneMap[K comparable, V any](source map[K]V) map[K]V {
	out := make(map[K]V, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}
