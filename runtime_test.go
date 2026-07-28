package steady

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

func testModel(t testing.TB) (*Model, []byte) {
	t.Helper()
	m := &Model{
		table: deterministicValues(64*8, 42), weights: deterministicValues(3*8, 43),
		bias: []float32{0, 0, 0}, temperature: 1,
		quantiles: []float32{1, 1, 1}, thresholds: []float32{0, 1, 0},
		bucket: 64, dim: 8, minN: 3, maxN: 5,
		metadata: Metadata{
			ArtifactFormat: 3, ModelID: "settings-v2", Task: "video-duration-selection",
			Labels: []string{"short", "medium", "long"}, PolicyCompatibility: ProfileQuotaSafeV2,
			MinN: 3, MaxN: 5, Bucket: 64, Dimension: 8, Epochs: 1,
			LearningRate: 0.05, L2: 0.0001, Alpha: 0.05, Seed: 42,
			SourceManifestSHA256: strings.Repeat("a", 64), TrainingCodeCommit: "test",
		},
	}
	data, err := marshalModel(m)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return loaded, data
}

func testV4Model(t testing.TB) (*Model, []byte) {
	t.Helper()
	const bucket = 64
	const dimension = 8
	m := &Model{
		table:            make([]float32, bucket*dimension),
		weights:          make([]float32, 2*(bucket+dimension+temporalFeatureCount)),
		bias:             []float32{5, -5},
		temperatures:     []float32{1, 1},
		quantiles:        []float32{1, 0, 1, 0},
		thresholds:       []float32{0.8, 0.8},
		bucket:           bucket,
		dim:              dimension,
		minN:             3,
		maxN:             5,
		wordMinN:         1,
		wordMaxN:         3,
		temporalFeatures: temporalFeatureCount,
		metadata: Metadata{
			ArtifactFormat: 4, ModelID: "settings-v2", Task: "video-duration-selection",
			Labels: []string{"short", "medium", "long"}, PolicyCompatibility: ProfileQuotaSafeV2,
			MinN: 3, MaxN: 5, Bucket: bucket, Dimension: dimension, Epochs: 1,
			LearningRate: 0.01, L2: 0.0001, Alpha: 0.05, Seed: 42,
			SourceManifestSHA256: strings.Repeat("c", 64), TrainingCodeCommit: "test-v4",
			ModelFamily: "dual-head-hybrid-v1", Heads: []string{"short", "long"},
			FeatureSchema: "hashed-char-word-linear+embedding+temporal-v1",
			WordMinN:      1, WordMaxN: 3, TemporalFeatures: temporalFeatureCount,
			PositiveClassWeights: []float64{1, 1},
		},
	}
	data, err := marshalModel(m)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	return loaded, data
}

func TestArtifactRoundTripAndDerivedIdentity(t *testing.T) {
	model, data := testModel(t)
	if model.Metadata().ArtifactSHA256 == "" || model.Metadata().ModelID != "settings-v2" {
		t.Fatalf("missing identity: %+v", model.Metadata())
	}
	path := t.TempDir() + "/model.bin"
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	reloaded, err := Load(path)
	if err != nil || reloaded.Metadata().ArtifactSHA256 != model.Metadata().ArtifactSHA256 {
		t.Fatalf("reload: %v %+v", err, reloaded)
	}
}

func TestV4ArtifactRoundTripAndDecision(t *testing.T) {
	model, data := testV4Model(t)
	if model.Metadata().ArtifactFormat != 4 || len(data) >= 32<<20 {
		t.Fatalf("unexpected v4 artifact: %+v bytes=%d", model.Metadata(), len(data))
	}
	result, err := PickSettings(model, QuotaSafeProfile(), PickRequest{
		Prompt: "one quick wink", Mode: TextToVideo,
	})
	if err != nil || result.Duration != 2 || result.DurationSource != "model" {
		t.Fatalf("v4 decision: %+v %v", result, err)
	}
}

func TestMarshalRejectsOversizedMetadata(t *testing.T) {
	model, _ := testModel(t)
	model.metadata.ArtifactSHA256 = ""
	model.metadata.TrainingCodeCommit = strings.Repeat("x", maxMetadataSize)
	if _, err := marshalModel(model); err == nil ||
		!strings.Contains(err.Error(), "metadata is too large") {
		t.Fatalf("marshal oversized metadata error = %v", err)
	}
}

func TestEncoderSoftmaxAndConformalPrimitives(t *testing.T) {
	table := deterministicValues(32*8, 42)
	first := make([]float32, 8)
	second := make([]float32, 8)
	if encodeInto("Hello", table, 32, 8, 3, 5, first) == 0 {
		t.Fatal("encoder produced no n-grams")
	}
	encodeInto("hello", table, 32, 8, 3, 5, second)
	if !slices.Equal(first, second) {
		t.Fatal("ASCII case normalization is not deterministic")
	}
	probabilities := make([]float32, 3)
	softmax([]float32{1, 2, 3}, probabilities, 1)
	sum := probabilities[0] + probabilities[1] + probabilities[2]
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("softmax sum = %f", sum)
	}
	if got := conformalQuantile([]float32{0.1, 0.2, 0.3, 0.4}, 0.25); got != 0.4 {
		t.Fatalf("conformal quantile = %f", got)
	}
}

func TestArtifactRejectsMalformedAndLegacy(t *testing.T) {
	_, valid := testModel(t)
	tests := [][]byte{
		nil,
		valid[:20],
		append(slices.Clone(valid), 0),
	}
	legacy := make([]byte, 32)
	binary.LittleEndian.PutUint32(legacy[:4], 0x42595445)
	binary.LittleEndian.PutUint32(legacy[4:8], 2)
	tests = append(tests, legacy)
	nonFinite := slices.Clone(valid)
	metaLen := int(binary.LittleEndian.Uint32(nonFinite[8:12]))
	binary.LittleEndian.PutUint32(nonFinite[modelHeaderSize+metaLen:], math.Float32bits(float32(math.NaN())))
	tests = append(tests, nonFinite)
	for i, data := range tests {
		if _, err := LoadBytes(data); err == nil {
			t.Fatalf("case %d accepted", i)
		}
	}
}

func TestPolicyTable(t *testing.T) {
	profile := QuotaSafeProfile()
	tests := []struct {
		request    PickRequest
		duration   int
		aspect     string
		resolution string
		source     string
	}{
		{PickRequest{Prompt: "a cat walks", Mode: TextToVideo}, 4, "16:9", "480p", "fallback"},
		{PickRequest{Prompt: "show it for 3 seconds", Mode: TextToVideo}, 4, "16:9", "480p", "explicit"},
		{PickRequest{Prompt: "show it for 12 seconds", Mode: TextToVideo}, 6, "16:9", "480p", "explicit"},
		{PickRequest{Prompt: "not 2 seconds, a normal shot", Mode: TextToVideo}, 4, "16:9", "480p", "fallback"},
		{PickRequest{Prompt: "2 seconds then 6 seconds", Mode: TextToVideo}, 4, "16:9", "480p", "fallback"},
		{PickRequest{Prompt: "9:16 at 720p", Mode: TextToVideo}, 4, "9:16", "720p", "fallback"},
		{PickRequest{Prompt: "make it 1:1", Mode: ImageToVideo, SourceMediaAspectRatio: "4:3"}, 4, "auto", "480p", "fallback"},
		{PickRequest{Prompt: "make it", Mode: ImageToVideo, SourceMediaAspectRatio: "4:3", AspectRatio: "1:1"}, 4, "1:1", "480p", "fallback"},
	}
	for _, test := range tests {
		got, err := PickSettings(nil, profile, test.request)
		if err != nil {
			t.Fatal(err)
		}
		if got.Duration != test.duration || got.AspectRatio != test.aspect ||
			got.Resolution != test.resolution || got.Source != test.source {
			t.Fatalf("%q: %+v", test.request.Prompt, got)
		}
	}
}

func TestReasonsEncodeAsArrayAndReturnOwnedStrings(t *testing.T) {
	result, err := PickSettings(nil, QuotaSafeProfile(), PickRequest{
		Prompt: "a cat walks", Mode: TextToVideo,
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(data, []byte(`"reasons":["duration.safe_fallback"`)) {
		t.Fatalf("reasons are not a JSON array: %s", data)
	}
	first := result.Reasons.Strings()
	second := result.Reasons.Strings()
	first[0] = "mutated"
	if second[0] == "mutated" {
		t.Fatal("Reasons.Strings returned shared storage")
	}
}

func TestCustomProfileAndValidation(t *testing.T) {
	cfg := QuotaSafeProfile().config
	cfg.Name, cfg.Version = "full", "1"
	cfg.AllowedDurations = []int{1, 5, 10, 15}
	cfg.SemanticDurations = map[string]int{"short": 1, "medium": 5, "long": 10}
	cfg.DefaultDuration, cfg.MaximumDuration = 5, 15
	profile, err := NewProfile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	got, err := PickSettings(nil, profile, PickRequest{Prompt: "14 seconds", Mode: TextToVideo})
	if err != nil || got.Duration != 15 {
		t.Fatalf("custom: %+v %v", got, err)
	}
	if _, err := PickSettings(nil, profile, PickRequest{Prompt: string([]byte{0xff}), Mode: TextToVideo}); err == nil {
		t.Fatal("invalid UTF-8 accepted")
	}
}

func TestConcurrentSharedModel(t *testing.T) {
	model, _ := testModel(t)
	const goroutines = 100
	const each = 1000
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for worker := 0; worker < goroutines; worker++ {
		go func() {
			defer wg.Done()
			for i := 0; i < each; i++ {
				result, err := PickSettings(model, QuotaSafeProfile(), PickRequest{
					Prompt: "a simple natural action", Mode: TextToVideo,
				})
				if err != nil || len(result.ArtifactSHA256) != 64 {
					t.Errorf("prediction failed: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestConcurrentSharedV4Model(t *testing.T) {
	model, _ := testV4Model(t)
	const goroutines = 100
	const each = 1000
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for worker := 0; worker < goroutines; worker++ {
		go func() {
			defer wg.Done()
			for i := 0; i < each; i++ {
				result, err := PickSettings(model, QuotaSafeProfile(), PickRequest{
					Prompt: "one quick wink", Mode: TextToVideo,
				})
				if err != nil || result.Duration != 2 {
					t.Errorf("v4 prediction failed: %+v %v", result, err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestTrainingIsByteDeterministic(t *testing.T) {
	dir := t.TempDir()
	rows := []byte(
		"__label__short one quick wink\n" +
			"__label__short one brief flash\n" +
			"__label__medium a person walks across a room\n" +
			"__label__medium waves move onto a shore\n" +
			"__label__long a seed sprouts then grows and blooms\n" +
			"__label__long a house progresses through construction stages\n",
	)
	split := filepath.Join(dir, "split.txt")
	if err := os.WriteFile(split, rows, 0o600); err != nil {
		t.Fatal(err)
	}
	run := func(name string) []byte {
		cfg := DefaultTrainConfig()
		cfg.TrainInput, cfg.ProbabilityCalibrationInput = split, split
		cfg.ConformalCalibrationInput, cfg.ThresholdDevelopmentInput = split, split
		cfg.Output = filepath.Join(dir, name)
		cfg.Bucket, cfg.Dimension, cfg.Epochs = 64, 8, 2
		cfg.SourceManifestSHA256 = strings.Repeat("b", 64)
		cfg.TrainingCodeCommit = "fixed-test-commit"
		if err := Train(cfg); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(cfg.Output)
		if err != nil {
			t.Fatal(err)
		}
		model, err := LoadBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		if model.Metadata().ArtifactFormat != 4 ||
			model.Metadata().ModelFamily != "dual-head-hybrid-v1" {
			t.Fatalf("unexpected trained metadata: %+v", model.Metadata())
		}
		return data
	}
	if first, second := run("first.bin"), run("second.bin"); !bytes.Equal(first, second) {
		t.Fatal("identical training runs produced different model bytes")
	}
}

func TestDefaultModelIsV5AndSelfContained(t *testing.T) {
	model, err := LoadDefault()
	if err != nil {
		t.Fatal(err)
	}
	if model.Metadata().ArtifactFormat != 5 ||
		model.Metadata().DecisionPolicy != DecisionPolicyLongOnlyPragmaticV2 ||
		len(model.Metadata().ArtifactSHA256) != 64 {
		t.Fatalf("unexpected embedded metadata: %+v", model.Metadata())
	}
	if len(defaultModelArtifact) >= 10<<20 {
		t.Fatalf("embedded artifact is %d bytes, want less than 10 MiB", len(defaultModelArtifact))
	}
}

func TestCLIHealthAndNDJSON(t *testing.T) {
	health := exec.Command("go", "run", "./cmd/steady-picker", "health")
	healthOutput, err := health.Output()
	if err != nil || !bytes.Contains(healthOutput, []byte(`"status":"ok"`)) ||
		!bytes.Contains(healthOutput, []byte(`"ready":true`)) ||
		!bytes.Contains(healthOutput, []byte(
			`"decision_policy":"long-only-pragmatic-v2"`,
		)) {
		t.Fatalf("health failed: %v %s", err, healthOutput)
	}
	command := exec.Command("go", "run", "./cmd/steady-picker", "predict")
	command.Stdin = strings.NewReader(
		"{\"prompt\":\"a cat walks\",\"mode\":\"text-to-video\"}\n" +
			"{\"prompt\":\"3 seconds at 720p\",\"mode\":\"text-to-video\"}\n",
	)
	output, err := command.Output()
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(output), []byte{'\n'})
	if len(lines) != 2 {
		t.Fatalf("NDJSON output has %d rows", len(lines))
	}
	for _, line := range lines {
		var result PickResult
		if err := json.Unmarshal(line, &result); err != nil {
			t.Fatal(err)
		}
	}
}

func BenchmarkPickSettingsFallback(b *testing.B) {
	model, _ := testModel(b)
	profile := QuotaSafeProfile()
	request := PickRequest{Prompt: "a person walks naturally", Mode: TextToVideo}
	b.ReportAllocs()
	for range b.N {
		if _, err := PickSettings(model, profile, request); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPickSettingsV4(b *testing.B) {
	model, _ := testV4Model(b)
	profile := QuotaSafeProfile()
	request := PickRequest{
		Prompt: "A seed sprouts, grows into a tree, and finally blooms.",
		Mode:   TextToVideo,
	}
	_, _ = PickSettings(model, profile, request)
	b.ReportAllocs()
	for range b.N {
		if _, err := PickSettings(model, profile, request); err != nil {
			b.Fatal(err)
		}
	}
}

func FuzzLoadBytes(f *testing.F) {
	_, seed := testModel(f)
	f.Add(seed)
	v5Header := make([]byte, modelHeaderSize+2)
	binary.LittleEndian.PutUint32(v5Header[:4], modelMagic)
	binary.LittleEndian.PutUint32(v5Header[4:8], modelVersion)
	binary.LittleEndian.PutUint32(v5Header[8:12], 1)
	binary.LittleEndian.PutUint64(v5Header[16:24], 1)
	f.Add(v5Header)
	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = LoadBytes(data)
	})
}

func FuzzProfileJSON(f *testing.F) {
	f.Add([]byte(`{"name":"x","version":"1"}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var config ProfileConfig
		if json.Unmarshal(data, &config) == nil {
			_, _ = NewProfile(config)
		}
	})
}

func FuzzPickRequest(f *testing.F) {
	f.Add("a cat walks", "text-to-video")
	f.Fuzz(func(t *testing.T, prompt, mode string) {
		_, _ = PickSettings(nil, QuotaSafeProfile(), PickRequest{Prompt: prompt, Mode: Mode(mode)})
	})
}

func FuzzCueParsing(f *testing.F) {
	f.Add("not 2 seconds, use 16:9 at 720p")
	f.Fuzz(func(t *testing.T, prompt string) {
		_ = affirmativeDurationCues(prompt)
		_ = affirmativeCues(prompt, aspectPattern, normalizeAspect)
		_ = affirmativeCues(prompt, resolutionPattern, strings.ToLower)
	})
}
