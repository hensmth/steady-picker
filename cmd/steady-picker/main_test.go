package main

import (
	"strings"
	"testing"
)

func TestDecodePickRequest(t *testing.T) {
	request, err := decodePickRequest(strings.NewReader(
		`{"prompt":"a quick wink","mode":"text-to-video"}` + "\n",
	))
	if err != nil {
		t.Fatal(err)
	}
	if request.Prompt != "a quick wink" || request.Mode != "text-to-video" {
		t.Fatalf("unexpected request: %+v", request)
	}
}

func TestDecodePickRequestRejectsUnsafeInput(t *testing.T) {
	for _, input := range []string{
		"",
		"{}\n",
		`{"prompt":"test","mode":"other"}` + "\n",
		`{"prompt":"test","mode":"text-to-video","image_aspect_ratio":-1}` + "\n",
		`{"prompt":"one","mode":"text-to-video"}` + "\n" +
			`{"prompt":"two","mode":"text-to-video"}` + "\n",
	} {
		if _, err := decodePickRequest(strings.NewReader(input)); err == nil {
			t.Fatalf("expected input to fail: %q", input)
		}
	}
}

func TestBuildVersionPrefersInjectedReleaseVersion(t *testing.T) {
	previous := version
	version = "v9.9.9"
	t.Cleanup(func() { version = previous })
	if got := buildVersion(); got != "v9.9.9" {
		t.Fatalf("buildVersion() = %q", got)
	}
}
