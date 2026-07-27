package steady

import (
	"testing"

	"github.com/xDarkicex/memory"
)

func testPool(t *testing.T) *memory.Pool {
	t.Helper()
	p, err := memory.NewPool(memory.AllocatorConfig{PoolSize: 1024 * 1024, SlabSize: 4096, SlabCount: 2}, 64)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Free() })
	return p
}

func TestPredictionSetEmpty(t *testing.T) {
	ps := PredictionSet{}
	if !ps.IsEmpty() {
		t.Fatal("empty PredictionSet should be empty")
	}
	ps = PredictionSet{Kinds: []string{"identity"}}
	if ps.IsEmpty() {
		t.Fatal("non-empty PredictionSet should not be empty")
	}
}

func TestComputeQuantile(t *testing.T) {
	// Perfect classifier: all scores are 0 (no error). Q should be 0.
	scores := []float32{0, 0, 0, 0, 0}
	q := ComputeQuantile(scores, 0.05)
	if q != 0 {
		t.Fatalf("perfect classifier Q = %f, want 0", q)
	}
}

func TestComputeQuantileEmpty(t *testing.T) {
	q := ComputeQuantile(nil, 0.05)
	if q != 1.0 {
		t.Fatalf("empty scores Q = %f, want 1.0", q)
	}
}

func TestPredictSetAllIncluded(t *testing.T) {
	pool := testPool(t)
	probs := []float32{0.999, 0.998, 0.997}
	names := []string{"a", "b", "c"}
	ps := PredictSet(probs, names, 0.01, pool)
	if ps.IsEmpty() {
		t.Fatal("expected non-empty set")
	}
	if len(ps.Kinds) != 3 {
		t.Fatalf("expected 3 kinds, got %d", len(ps.Kinds))
	}
}

func TestPredictSetAllExcluded(t *testing.T) {
	pool := testPool(t)
	probs := []float32{0.1, 0.1, 0.1}
	names := []string{"a", "b", "c"}
	ps := PredictSet(probs, names, 0.01, pool)
	if !ps.IsEmpty() {
		t.Fatal("expected empty set for noise")
	}
}

func TestPredictSetPartial(t *testing.T) {
	probs := []float32{0.999, 0.95, 0.8}
	names := []string{"a", "b", "c"}
	// Q = 0.1 → include if (1-p) <= 0.1
	// a: 0.001 <= 0.1 ✓
	// b: 0.05 <= 0.1 ✓
	// c: 0.2 > 0.1 ✗
	pool := testPool(t)
	ps := PredictSet(probs, names, 0.1, pool)
	if len(ps.Kinds) != 2 {
		t.Fatalf("expected 2 kinds, got %d", len(ps.Kinds))
	}
}

func BenchmarkPredictSet(b *testing.B) {
	pool, _ := memory.NewPool(memory.AllocatorConfig{PoolSize: 1024 * 1024, SlabSize: 4096, SlabCount: 2}, 64)
	defer pool.Free()
	probs := []float32{0.95, 0.90, 0.85, 0.80, 0.10, 0.10}
	names := []string{"identity", "constraint", "decision", "fact", "preference", "episode"}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		PredictSet(probs, names, 0.1, pool)
		pool.Reset()
	}
}

func BenchmarkComputeQuantile(b *testing.B) {
	scores := make([]float32, 1000)
	for i := range scores {
		scores[i] = float32(i) / 1000.0
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		ComputeQuantile(scores, 0.05)
	}
}
