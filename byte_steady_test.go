package steady

import (
	"testing"

	"github.com/xDarkicex/memory"
)

func TestEncodeEmpty(t *testing.T) {
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  1024 * 1024,
		SlabSize:  4096,
		SlabCount: 2,
	}, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()

	table := makeTable(1024, 8)
	vec := Encode(nil, table, 1024, 8, pool)
	if len(vec) != 8 {
		t.Fatalf("expected dim=8, got %d", len(vec))
	}
	for i, v := range vec {
		if v != 0 {
			t.Fatalf("expected zero vector for empty input, got vec[%d]=%f", i, v)
		}
	}
}

func TestEncodeShortText(t *testing.T) {
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  1024 * 1024,
		SlabSize:  4096,
		SlabCount: 2,
	}, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()

	table := makeTable(1024, 8)
	// Text shorter than smallest window (4 bytes)
	vec := Encode([]byte("hi"), table, 1024, 8, pool)
	if len(vec) != 8 {
		t.Fatalf("expected dim=8, got %d", len(vec))
	}
	for i, v := range vec {
		if v != 0 {
			t.Fatalf("expected zero vector for sub-window input, got vec[%d]=%f", i, v)
		}
	}
}

func TestEncodeASCII(t *testing.T) {
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  1024 * 1024,
		SlabSize:  4096,
		SlabCount: 2,
	}, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()

	bucket, dim := 1024, 8
	table := makeTable(bucket, dim)
	text := []byte("I am a Go developer")
	vec := Encode(text, table, bucket, dim, pool)

	if len(vec) != dim {
		t.Fatalf("expected dim=%d, got %d", dim, len(vec))
	}
	// Vector should be non-zero for non-empty text longer than min window
	nonZero := false
	for _, v := range vec {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected non-zero vector for valid text")
	}
}

func TestEncodeUTF8(t *testing.T) {
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  1024 * 1024,
		SlabSize:  4096,
		SlabCount: 2,
	}, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()

	bucket, dim := 1024, 8
	table := makeTable(bucket, dim)
	text := []byte("こんにちは世界") // Japanese UTF-8
	vec := Encode(text, table, bucket, dim, pool)

	if len(vec) != dim {
		t.Fatalf("expected dim=%d, got %d", dim, len(vec))
	}
}

func TestHashWindowDeterminism(t *testing.T) {
	data := []byte("hello world")
	h1 := hashWindow(data)
	h2 := hashWindow(data)
	if h1 != h2 {
		t.Fatal("hashWindow not deterministic")
	}
}

func TestHashWindowSameContent(t *testing.T) {
	// Same content should produce same hash regardless of surrounding context
	h1 := hashWindow([]byte("test"))
	h2 := hashWindow([]byte("test"))
	if h1 != h2 {
		t.Fatal("hashWindow not consistent for same content")
	}
}

func TestEncodeBenchmark(t *testing.T) {
	// Smoke test for the benchmark path — 50-word chat message
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  1024 * 1024,
		SlabSize:  4096,
		SlabCount: 2,
	}, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()

	bucket, dim := 20000, 64
	table := makeTable(bucket, dim)
	text := []byte("I am a Go developer working on distributed systems and I prefer Neovim over VS Code")
	vec := Encode(text, table, bucket, dim, pool)

	if len(vec) != dim {
		t.Fatalf("expected dim=%d, got %d", dim, len(vec))
	}
}

// makeTable creates a synthetic embedding table for testing.
// Each row i gets a distinct pattern so different hashes produce different accumulations.
func makeTable(bucket, dim int) []float32 {
	table := make([]float32, bucket*dim)
	for i := range bucket {
		for j := range dim {
			table[i*dim+j] = float32(i+1) * 0.01 * float32(j+1)
		}
	}
	return table
}

func BenchmarkEncode(b *testing.B) {
	pool, _ := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  64 * 1024 * 1024,
		SlabSize:  1024 * 1024,
		SlabCount: 16,
	}, 64)
	defer pool.Free()

	bucket, dim := 20000, 64
	table := makeTable(bucket, dim)
	text := []byte("I am a Go developer working on distributed systems and I prefer Neovim over VS Code")

	b.ResetTimer()
	b.ReportAllocs()
	for b.Loop() {
		Encode(text, table, bucket, dim, pool)
		pool.Reset()
	}
}
