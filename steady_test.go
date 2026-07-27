package steady

import (
	"testing"

	"github.com/xDarkicex/memory"
)

func TestClassifyNilModel(t *testing.T) {
	var m *Model
	ps := m.Classify("hello")
	if !ps.IsEmpty() {
		t.Fatal("nil model should return empty set")
	}
}

func TestClassifyEmptyText(t *testing.T) {
	m := &Model{
		bucket:     1024,
		dim:        16,
		numLabels:  2,
		labelNames: []string{"a", "b"},
	}
	ps := m.Classify("")
	if !ps.IsEmpty() {
		t.Fatal("expected empty set when pool is nil")
	}
	// With no pool, Encode returns nil slice, so Classify returns empty set.
}

func BenchmarkClassify(b *testing.B) {
	// Synthetic model for benchmark
	bucket, dim := 1024, 32
	numLabels := 6
	pool, _ := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  64 * 1024 * 1024,
		SlabSize:  1024 * 1024,
		SlabCount: 16,
	}, 64)
	defer pool.Free()

	table := makeTable(bucket, dim)
	weights := memory.MustPoolSlice[float32](pool, numLabels*dim)
	weights = weights[:numLabels*dim]
	for i := range weights {
		weights[i] = 0.01
	}
	bias := memory.MustPoolSlice[float32](pool, numLabels)
	bias = bias[:numLabels]
	plattA := memory.MustPoolSlice[float32](pool, numLabels)
	plattA = plattA[:numLabels]
	plattB := memory.MustPoolSlice[float32](pool, numLabels)
	plattB = plattB[:numLabels]
	for i := range plattA {
		plattA[i] = 1.0
	}

	m := &Model{
		table:       table,
		weights:     weights,
		bias:        bias,
		plattA:      plattA,
		plattB:      plattB,
		q:           0.5,
		bucket:      bucket,
		dim:         dim,
		numLabels:   numLabels,
		labelNames:  []string{"identity", "constraint", "decision", "fact", "preference", "episode"},
		modelPool:   pool,
		scratchPool: pool,
	}

	text := "I am a Go developer and API keys must never be committed"
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		m.Classify(text)
	}
}
