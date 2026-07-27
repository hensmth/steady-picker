package steady

import "testing"

func TestSigmoid(t *testing.T) {
	if sigmoid(0) != 0.5 {
		t.Fatalf("sigmoid(0) = %f, want 0.5", sigmoid(0))
	}
	if sigmoid(10) <= 0.9 {
		t.Fatalf("sigmoid(10) too low: %f", sigmoid(10))
	}
	if sigmoid(-10) >= 0.1 {
		t.Fatalf("sigmoid(-10) too high: %f", sigmoid(-10))
	}
}

func TestPredictLogitsAllZero(t *testing.T) {
	numLabels := 3
	hidden := []float32{0, 0, 0, 0}
	weights := make([]float32, numLabels*len(hidden))
	bias := make([]float32, numLabels)
	out := make([]float32, numLabels)

	PredictLogits(hidden, weights, bias, out)
	// sigmoid(0) = 0.5 for every label when weights and bias are all zero
	for i, v := range out {
		if v != 0.5 {
			t.Fatalf("label %d: expected 0.5, got %f", i, v)
		}
	}
}

func TestPredictLogitsKnownWeights(t *testing.T) {
	numLabels := 2
	hidden := []float32{1.0, 0.0}
	// Label 0: dot([1,0], [1,0]) + 0 = 1 → sigmoid(1) ≈ 0.731
	// Label 1: dot([1,0], [0,1]) + 0 = 0 → sigmoid(0) = 0.5
	weights := []float32{1, 0, 0, 1}
	bias := []float32{0, 0}
	out := make([]float32, numLabels)

	PredictLogits(hidden, weights, bias, out)

	expected0 := float32(0.7310586)
	if out[0] < expected0-0.01 || out[0] > expected0+0.01 {
		t.Fatalf("label 0: expected ~%f, got %f", expected0, out[0])
	}
	if out[1] != 0.5 {
		t.Fatalf("label 1: expected 0.5, got %f", out[1])
	}
}

func BenchmarkPredictLogits(b *testing.B) {
	dim, numLabels := 64, 6
	hidden := make([]float32, dim)
	for i := range hidden {
		hidden[i] = float32(i) * 0.01
	}
	weights := make([]float32, numLabels*len(hidden))
	bias := make([]float32, numLabels)
	out := make([]float32, numLabels)

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		PredictLogits(hidden, weights, bias, out)
	}
}
