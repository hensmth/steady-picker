package steady

import (
	"testing"

	"github.com/xDarkicex/memory"
)

func testPlattPool(t *testing.T) *memory.Pool {
	t.Helper()
	p, err := memory.NewPool(memory.AllocatorConfig{PoolSize: 1024 * 1024, SlabSize: 4096, SlabCount: 2}, 64)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Free() })
	return p
}

func TestApplyPlatt(t *testing.T) {
	// Identity: A=1, B=0 should pass through score=0 → 0.5
	p := ApplyPlatt(0, 1, 0)
	if p != 0.5 {
		t.Fatalf("Platt(0, 1, 0) = %f, want 0.5", p)
	}
	// Score=0, A=0, B=0 → sigmoid(0) = 0.5
	p = ApplyPlatt(0, 0, 0)
	if p != 0.5 {
		t.Fatalf("Platt(0, 0, 0) = %f, want 0.5", p)
	}
}

func TestCalibratePlatt(t *testing.T) {
	// Perfect classifier: raw scores are 1.0 for correct label, 0 for others.
	scores := [][]float32{
		{1.0, 0.0},
		{1.0, 0.0},
		{1.0, 0.0},
		{0.0, 1.0},
		{0.0, 1.0},
		{0.0, 1.0},
	}
	labels := []int{0, 0, 0, 1, 1, 1}
	a, b := CalibratePlatt(testPlattPool(t), scores, labels, 2)
	// A should be positive (increasing confidence for correct label)
	if a[0] <= 0 || a[1] <= 0 {
		t.Fatalf("expected positive A, got A0=%f A1=%f", a[0], a[1])
	}
	_ = b
}

func TestCalibratePlattEmpty(t *testing.T) {
	a, b := CalibratePlatt(testPlattPool(t), nil, nil, 0)
	if a != nil || b != nil {
		t.Fatal("expected nil for empty input")
	}
}

func BenchmarkApplyPlatt(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		ApplyPlatt(0.5, 0.1, 0.2)
	}
}
