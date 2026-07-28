package steady

import (
	"strings"
	"sync"
	"testing"
)

func TestSemanticV5RoundTripAndDecisions(t *testing.T) {
	shortArtifact := semanticTestArtifact(t, []float32{8, -8})
	if len(shortArtifact) >= 32<<20 {
		t.Fatalf("v5 test artifact is too large: %d bytes", len(shortArtifact))
	}
	shortModel, err := LoadBytes(shortArtifact)
	if err != nil {
		t.Fatal(err)
	}
	if got := shortModel.Metadata().ArtifactFormat; got != 5 {
		t.Fatalf("artifact format = %d, want 5", got)
	}
	short := shortModel.Classify("A bird hops once.")
	if len(short.Kinds) != 1 || short.Kinds[0] != "short" {
		t.Fatalf("short decision = %#v", short)
	}

	longArtifact := semanticTestArtifact(t, []float32{-8, 8})
	longModel, err := LoadBytes(longArtifact)
	if err != nil {
		t.Fatal(err)
	}
	long := longModel.Classify("First a seed sprouts, then it blooms, and finally wilts.")
	if len(long.Kinds) != 1 || long.Kinds[0] != "long" {
		t.Fatalf("long decision = %#v", long)
	}
	fallback := longModel.Classify("A landscape.")
	if len(fallback.Kinds) != 1 || fallback.Kinds[0] != "medium" {
		t.Fatalf("cue-ineligible long decision = %#v", fallback)
	}
}

func TestSemanticV5RejectsMalformedPayload(t *testing.T) {
	artifact := semanticTestArtifact(t, []float32{8, -8})
	if _, err := LoadBytes(artifact[:len(artifact)-1]); err == nil ||
		!strings.Contains(err.Error(), "payload") {
		t.Fatalf("truncated v5 error = %v", err)
	}
}

func TestSemanticV5MetadataIsCallerOwned(t *testing.T) {
	model, err := LoadBytes(semanticTestArtifact(t, []float32{8, -8}))
	if err != nil {
		t.Fatal(err)
	}
	metadata := model.Metadata()
	metadata.Vocabulary[0] = "changed"
	metadata.AuxiliaryHeads[0] = "changed"
	got := model.Metadata()
	if got.Vocabulary[0] != "[PAD]" || got.AuxiliaryHeads[0] != "beats_1" {
		t.Fatal("Metadata returned model-owned semantic slices")
	}
}

func TestSemanticV5WordPieceFailureReplacesWholeWord(t *testing.T) {
	model, err := LoadBytes(semanticTestArtifact(t, []float32{8, -8}))
	if err != nil {
		t.Fatal(err)
	}
	work := model.newWorkspace()
	count := model.semantic.tokenize("azzz", work)
	got := work.tokenIDs[:count]
	want := []int{2, 1, 3}
	if len(got) != len(want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("tokens = %v, want %v", got, want)
		}
	}
}

func TestSemanticV5SharedModel100000Predictions(t *testing.T) {
	model, err := LoadBytes(semanticTestArtifact(t, []float32{8, -8}))
	if err != nil {
		t.Fatal(err)
	}
	const workers = 100
	const perWorker = 1000
	var wait sync.WaitGroup
	wait.Add(workers)
	for range workers {
		go func() {
			defer wait.Done()
			for range perWorker {
				result := model.Classify("a")
				if len(result.Kinds) != 1 || result.Kinds[0] != "short" {
					t.Errorf("concurrent v5 decision = %#v", result)
					return
				}
			}
		}()
	}
	wait.Wait()
}

func BenchmarkSemanticV5Prediction(b *testing.B) {
	artifact := semanticTestArtifact(b, []float32{8, -8})
	model, err := LoadBytes(artifact)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, ok := model.pickDecision("a bird hops once"); !ok {
			b.Fatal("semantic decision was not accepted")
		}
	}
}

func semanticTestArtifact(t testing.TB, durationBias []float32) []byte {
	t.Helper()
	vocabulary := make([]string, semanticVocabSize)
	copy(vocabulary, []string{
		"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
		"a", "bird", "hop", "##s", "once", ".", "first", "seed", "sprout",
		"then", "it", "bloom", "finally", "wilt", "landscape",
	})
	for index := 20; index < len(vocabulary); index++ {
		vocabulary[index] = "[unused-" + fixedDecimal(index) + "]"
	}
	metadata := Metadata{
		ArtifactFormat:       5,
		ModelID:              "settings-v2",
		Task:                 "video-duration-selection",
		Labels:               []string{"short", "medium", "long"},
		PolicyCompatibility:  "quota-safe-v2",
		MinN:                 1,
		MaxN:                 1,
		Bucket:               1,
		Dimension:            semanticHidden,
		Epochs:               1,
		LearningRate:         0.001,
		L2:                   0.01,
		Alpha:                0.1,
		Seed:                 42,
		SourceManifestSHA256: strings.Repeat("0", 64),
		TrainingCodeCommit:   "test",
		ModelFamily:          "distilled-semantic-transformer-v1",
		FeatureSchema:        "wordpiece-transformer-temporal-aux-v1",
		Heads:                []string{"short", "long"},
		PositiveClassWeights: []float64{1, 1},
		Tokenizer:            "wordpiece-v1",
		Vocabulary:           vocabulary,
		MaxTokens:            semanticMaxTokens,
		Layers:               semanticLayers,
		AttentionHeads:       semanticHeads,
		HiddenSize:           semanticHidden,
		IntermediateSize:     semanticFFN,
		Quantization:         "int8-per-output-channel-f32-layernorm",
		AuxiliaryHeads: []string{
			"beats_1", "beats_2", "beats_3",
			"ordered", "transformation", "dependent_actions",
		},
		TeacherEncoder:   "sentence-transformers/paraphrase-MiniLM-L3-v2",
		TrainingProvider: "openai-codex",
		TrainingModel:    "gpt-5.6-sol",
		TrainingEffort:   "ultra",
	}
	zeroMatrix := func(rows, columns int, bias bool) quantizedMatrix {
		matrix := quantizedMatrix{
			rows: rows, cols: columns,
			weights: make([]int8, rows*columns),
			scales:  make([]float32, rows),
		}
		if bias {
			matrix.bias = make([]float32, rows)
		}
		return matrix
	}
	norm := func() layerNormWeights {
		gamma := make([]float32, semanticHidden)
		for index := range gamma {
			gamma[index] = 1
		}
		return layerNormWeights{
			gamma: gamma,
			beta:  make([]float32, semanticHidden),
		}
	}
	semantic := &semanticModel{
		tokenEmbeddings:    zeroMatrix(semanticVocabSize, semanticHidden, false),
		positionEmbeddings: zeroMatrix(semanticMaxTokens, semanticHidden, false),
		embeddingNorm:      norm(),
		vocabulary:         vocabulary,
		maxTokens:          semanticMaxTokens,
		hidden:             semanticHidden,
		attentionHeads:     semanticHeads,
		intermediate:       semanticFFN,
		layers:             make([]semanticLayer, semanticLayers),
	}
	for index := range semantic.layers {
		semantic.layers[index] = semanticLayer{
			query:           zeroMatrix(semanticHidden, semanticHidden, true),
			key:             zeroMatrix(semanticHidden, semanticHidden, true),
			value:           zeroMatrix(semanticHidden, semanticHidden, true),
			attentionOutput: zeroMatrix(semanticHidden, semanticHidden, true),
			attentionNorm:   norm(),
			feedForwardIn:   zeroMatrix(semanticFFN, semanticHidden, true),
			feedForwardOut:  zeroMatrix(semanticHidden, semanticFFN, true),
			outputNorm:      norm(),
		}
	}
	semantic.durationHeads = zeroMatrix(2, semanticHidden, true)
	copy(semantic.durationHeads.bias, durationBias)
	semantic.auxiliaryHeads = zeroMatrix(semanticAuxHeads, semanticHidden, true)
	model := &Model{
		metadata:     metadata,
		semantic:     semantic,
		temperatures: []float32{1, 1},
		quantiles:    []float32{0.1, 0.1, 0.1, 0.1},
		thresholds:   []float32{0.8, 0.8},
	}
	artifact, err := marshalSemanticModel(model)
	if err != nil {
		t.Fatal(err)
	}
	return artifact
}

func fixedDecimal(value int) string {
	const digits = "0123456789"
	var buffer [5]byte
	for index := len(buffer) - 1; index >= 0; index-- {
		buffer[index] = digits[value%10]
		value /= 10
	}
	return string(buffer[:])
}
