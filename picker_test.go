package steady

import "testing"

func TestSafeFallback(t *testing.T) {
	got := PickSettings(nil, PickRequest{Prompt: "a cat walks", Mode: "text-to-video"}, "none")
	if got.Duration != 4 || got.AspectRatio != "16:9" || got.Resolution != "480p" {
		t.Fatalf("unexpected fallback: %+v", got)
	}
}

func TestExplicitPolicy(t *testing.T) {
	got := PickSettings(nil, PickRequest{
		Prompt: "render in HD as a vertical video for 12 seconds",
		Mode:   "text-to-video",
	}, "none")
	if got.Duration != 6 || got.AspectRatio != "9:16" || got.Resolution != "720p" {
		t.Fatalf("unexpected explicit decision: %+v", got)
	}
}

func TestDecorativeQualityStays480p(t *testing.T) {
	got := PickSettings(nil, PickRequest{
		Prompt: "cinematic 4K style, extremely high quality",
		Mode:   "text-to-video",
	}, "none")
	if got.Resolution != "480p" {
		t.Fatalf("decorative quality unlocked HD: %+v", got)
	}
}

func TestImageKeepsAutomaticAspect(t *testing.T) {
	got := PickSettings(nil, PickRequest{
		Prompt: "square video", Mode: "image-to-video", ImageAspectRatio: 0.75,
	}, "none")
	if got.AspectRatio != "auto" {
		t.Fatalf("image aspect was overridden: %+v", got)
	}
}

func TestDecodeDuration(t *testing.T) {
	duration, ok := decodeDuration("d6")
	if !ok || duration != 6 {
		t.Fatalf("decode failed: %d %t", duration, ok)
	}
}

func TestDurationEvidenceRequiresConservativeCue(t *testing.T) {
	for _, test := range []struct {
		prompt   string
		duration int
		want     bool
	}{
		{"a quick wink", 2, true},
		{"a portrait", 2, false},
		{"a normal action", 4, false},
		{"a flower transformation", 6, true},
		{"a flower grows", 6, false},
	} {
		if got := hasDurationEvidence(test.prompt, test.duration); got != test.want {
			t.Fatalf("hasDurationEvidence(%q, %d) = %v, want %v", test.prompt, test.duration, got, test.want)
		}
	}
}
