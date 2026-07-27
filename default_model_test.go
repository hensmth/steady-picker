package steady

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

const expectedDefaultModelSHA256 = "0358b5262d27288b7c05cbfb40ba9de6d8771f0d7f0ac085e27e3561672c9b4d"

func TestDefaultModelChecksum(t *testing.T) {
	sum := sha256.Sum256(defaultModelArtifact)
	if got := hex.EncodeToString(sum[:]); got != expectedDefaultModelSHA256 {
		t.Fatalf("default model checksum = %s, want %s", got, expectedDefaultModelSHA256)
	}
}

func TestDefaultModelPredictsWithoutExternalFile(t *testing.T) {
	model, err := LoadDefault()
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	got := PickSettings(model, PickRequest{
		Prompt: "a flower transforming in a timelapse",
		Mode:   "text-to-video",
	}, DefaultModelVersion)
	if got.Duration != 6 || got.Source != "model" {
		t.Fatalf("unexpected embedded-model decision: %+v", got)
	}
	if got.ModelVersion != DefaultModelVersion {
		t.Fatalf("model version = %q", got.ModelVersion)
	}
}

func TestLoadBytesRejectsMalformedModel(t *testing.T) {
	if _, err := LoadBytes([]byte("not a model")); err == nil {
		t.Fatal("expected malformed embedded model to fail")
	}
}
