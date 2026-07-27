package steady

import (
	"bytes"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/xDarkicex/memory"
)

func TestParseExamples(t *testing.T) {
	input := `__label__identity I am a Go developer
__label__constraint API keys must never be committed
__label__fact The sky is blue
# this is a comment
__label__decision I decided to use Redis
`
	tmp := t.TempDir() + "/test_data.txt"
	if err := os.WriteFile(tmp, []byte(input), 0644); err != nil {
		t.Fatal(err)
	}
	names := []string{"identity", "constraint", "decision", "fact", "preference", "episode"}
	examples, err := parseExamples(tmp, names)
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != 4 {
		t.Fatalf("expected 4 examples, got %d", len(examples))
	}
	if examples[0].label != 0 || examples[0].text != "I am a Go developer" {
		t.Fatalf("first example wrong: label=%d text=%q", examples[0].label, examples[0].text)
	}
	if examples[3].label != 2 || examples[3].text != "I decided to use Redis" {
		t.Fatalf("last example wrong: label=%d text=%q", examples[3].label, examples[3].text)
	}
}

func TestParseExamplesEmpty(t *testing.T) {
	tmp := t.TempDir() + "/empty.txt"
	os.WriteFile(tmp, []byte(""), 0644)
	examples, _ := parseExamples(tmp, []string{"a", "b"})
	if len(examples) != 0 {
		t.Fatalf("expected 0 examples, got %d", len(examples))
	}
}

func TestInitTable(t *testing.T) {
	table, err := initTable(100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(table) != 800 {
		t.Fatalf("expected 800 floats, got %d", len(table))
	}
	// Values should be small (initialized with scale 1/dim)
	for i, v := range table {
		if v > 1.0 || v < -1.0 {
			t.Fatalf("table[%d] = %f, expected small value", i, v)
		}
	}
	defer memory.Munmap(unsafeSliceBytes(table))
}

func TestDefaultTrainConfig(t *testing.T) {
	cfg := DefaultTrainConfig()
	if cfg.Bucket != 20000 {
		t.Fatalf("expected Bucket=20000, got %d", cfg.Bucket)
	}
	if cfg.Dim != 64 {
		t.Fatalf("expected Dim=64, got %d", cfg.Dim)
	}
	if cfg.Epochs != 25 {
		t.Fatalf("expected Epochs=25, got %d", cfg.Epochs)
	}
	if cfg.LR != 0.1 {
		t.Fatalf("expected LR=0.1, got %f", cfg.LR)
	}
}

func TestShuffle(t *testing.T) {
	examples := make([]example, 100)
	for i := range examples {
		examples[i] = example{text: "x", label: i % 6}
	}
	original := make([]example, 100)
	copy(original, examples)
	shuffle(examples)
	// After shuffle, the first element should rarely match
	sameCount := 0
	for i := range examples {
		if examples[i].label == original[i].label && examples[i].text == original[i].text {
			sameCount++
		}
	}
	// With 100 elements, probability of all matching is infinitesimal
	if sameCount > 90 {
		t.Fatalf("shuffle appears to not have randomized: %d/100 in same position", sameCount)
	}
}

func TestFastRand(t *testing.T) {
	seedRand(42)
	seen := make(map[uint64]bool)
	for i := 0; i < 1000; i++ {
		v := fastRand()
		if seen[v] {
			t.Fatalf("fastRand produced duplicate after %d calls", i)
		}
		seen[v] = true
	}
}

func TestAtomicAddFloat32(t *testing.T) {
	var wg sync.WaitGroup
	var val float32
	n := 1000
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < n; j++ {
				atomicAddFloat32(&val, 1.0)
			}
		}()
	}
	wg.Wait()
	expected := float32(4 * n)
	if val != expected {
		t.Fatalf("atomicAddFloat32: expected %f, got %f", expected, val)
	}
}

func TestAtomicAddFloat32Negative(t *testing.T) {
	var val float32 = 100.0
	atomicAddFloat32(&val, -50.0)
	if val != 50.0 {
		t.Fatalf("expected 50.0, got %f", val)
	}
}

func TestUnsafeSliceBytes(t *testing.T) {
	s := []float32{1.0, 2.0, 3.0}
	b := unsafeSliceBytes(s)
	if len(b) != 12 {
		t.Fatalf("expected 12 bytes, got %d", len(b))
	}
}

func TestTrainEndToEnd(t *testing.T) {
	// Generate small synthetic dataset with separable classes
	var lines []string
	// Identity patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__identity I am a software engineer")
		lines = append(lines, "__label__identity I work as a developer at Google")
	}
	// Constraint patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__constraint API keys must never be committed")
		lines = append(lines, "__label__constraint secrets shall not be shared")
	}
	// Decision patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__decision I decided to use PostgreSQL")
		lines = append(lines, "__label__decision let us use Redis for caching")
	}
	// Fact patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__fact the server runs on port 8080")
		lines = append(lines, "__label__fact the database uses PostgreSQL 16")
	}
	// Preference patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__preference I prefer Neovim over VS Code")
		lines = append(lines, "__label__preference I like Go more than Python")
	}
	// Episode patterns
	for i := 0; i < 50; i++ {
		lines = append(lines, "__label__episode yesterday I deployed to production")
		lines = append(lines, "__label__episode last week I went to a conference")
	}

	input := strings.Join(lines, "\n")
	tmpDir := t.TempDir()
	inputPath := tmpDir + "/train.txt"
	outputPath := tmpDir + "/model.bin"
	os.WriteFile(inputPath, []byte(input), 0644)

	cfg := DefaultTrainConfig()
	cfg.Input = inputPath
	cfg.Output = outputPath
	cfg.Bucket = 1000
	cfg.Dim = 8
	cfg.Epochs = 5
	cfg.LR = 0.1
	cfg.NumGoroutines = 2
	cfg.CalibSplit = 0.3
	cfg.Seed = 42
	cfg.LabelNames = []string{"identity", "constraint", "decision", "fact", "preference", "episode"}

	if err := Train(cfg); err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Load and verify
	m, err := Load(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	m.SetLabelNames(cfg.LabelNames)

	// Each class should classify its own text correctly
	tests := []struct {
		text     string
		expected string
	}{
		{"I am a software engineer", "identity"},
		{"API keys must never be committed", "constraint"},
		{"I decided to use PostgreSQL", "decision"},
		{"the server runs on port 8080", "fact"},
		{"I prefer Neovim over VS Code", "preference"},
		{"yesterday I deployed to production", "episode"},
	}
	correct := 0
	for _, tt := range tests {
		r := m.ClassifyDebug(tt.text)
		// Find highest-confidence label
		best, bestIdx := float32(0.0), 0
		for i, c := range r.Calibrated {
			if c > best {
				best = c
				bestIdx = i
			}
		}
		if cfg.LabelNames[bestIdx] == tt.expected {
			correct++
		}
	}
	if correct < 3 {
		t.Fatalf("only %d/6 correct, model not learning", correct)
	}
	t.Logf("End-to-end: %d/6 correct on synthetic data", correct)
}

func TestTrainIsDeterministicWithOneWorker(t *testing.T) {
	input := strings.Join([]string{
		"__label__a alpha one",
		"__label__a alpha two",
		"__label__a alpha three",
		"__label__a alpha four",
		"__label__b beta one",
		"__label__b beta two",
		"__label__b beta three",
		"__label__b beta four",
	}, "\n")
	dir := t.TempDir()
	inputPath := dir + "/train.txt"
	if err := os.WriteFile(inputPath, []byte(input), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultTrainConfig()
	cfg.Input = inputPath
	cfg.Bucket = 128
	cfg.Dim = 8
	cfg.Epochs = 3
	cfg.LabelNames = []string{"a", "b"}
	cfg.NumGoroutines = 1

	cfg.Output = dir + "/first.bin"
	if err := Train(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Output = dir + "/second.bin"
	if err := Train(cfg); err != nil {
		t.Fatal(err)
	}
	first, err := os.ReadFile(dir + "/first.bin")
	if err != nil {
		t.Fatal(err)
	}
	second, err := os.ReadFile(dir + "/second.bin")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("identical single-worker training runs produced different models")
	}
}

func TestSplitCalibrationIsStratifiedAndDisjoint(t *testing.T) {
	examples := []example{
		{text: "a1", label: 0}, {text: "b1", label: 1},
		{text: "a2", label: 0}, {text: "b2", label: 1},
		{text: "a3", label: 0}, {text: "b3", label: 1},
		{text: "a4", label: 0}, {text: "b4", label: 1},
	}
	train, calibration, err := splitCalibration(examples, 2, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if len(train) != 6 || len(calibration) != 2 {
		t.Fatalf("unexpected split sizes: train=%d calibration=%d", len(train), len(calibration))
	}
	seen := map[string]bool{}
	for _, ex := range train {
		seen[ex.text] = true
	}
	labels := map[int]int{}
	for _, ex := range calibration {
		if seen[ex.text] {
			t.Fatalf("%q occurs in training and calibration", ex.text)
		}
		labels[ex.label]++
	}
	if labels[0] != 1 || labels[1] != 1 {
		t.Fatalf("calibration is not stratified: %v", labels)
	}
}
