package steady

import (
	"encoding/binary"
	"os"
	"testing"
)

func TestSaveAndLoad(t *testing.T) {
	// Write a minimal model file.
	path := t.TempDir() + "/test_model.bin"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	bucket, dim, numLabels := 32, 8, 2
	table := make([]float32, bucket*dim)
	for i := range table {
		table[i] = float32(i) * 0.01
	}

	// Write header
	var hdr [32]byte
	binary.LittleEndian.PutUint32(hdr[0:4], modelMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], modelVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(bucket))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(dim))
	binary.LittleEndian.PutUint32(hdr[16:20], 4)  // minN
	binary.LittleEndian.PutUint32(hdr[20:24], 16) // maxN
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(numLabels))
	f.Write(hdr[:])

	// Write embedding table
	binary.Write(f, binary.LittleEndian, table)

	// Write OVA weights (numLabels × dim)
	weights := make([]float32, numLabels*dim)
	for i := range weights {
		weights[i] = 0.1
	}
	binary.Write(f, binary.LittleEndian, weights)

	// Write bias
	bias := make([]float32, numLabels)
	binary.Write(f, binary.LittleEndian, bias)

	// Write Platt A
	plattA := []float32{1.0, 1.0}
	binary.Write(f, binary.LittleEndian, plattA)

	// Write Platt B
	plattB := []float32{0.0, 0.0}
	binary.Write(f, binary.LittleEndian, plattB)

	// Write conformal quantile
	var q float32 = 0.1
	binary.Write(f, binary.LittleEndian, q)
	f.Close()

	// Load it back
	m, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	if m.bucket != bucket {
		t.Fatalf("bucket: want %d, got %d", bucket, m.bucket)
	}
	if m.dim != dim {
		t.Fatalf("dim: want %d, got %d", dim, m.dim)
	}
	if m.numLabels != numLabels {
		t.Fatalf("numLabels: want %d, got %d", numLabels, m.numLabels)
	}
	if m.q != q {
		t.Fatalf("q: want %f, got %f", q, m.q)
	}
	// Spot-check table
	if m.table[0] != table[0] {
		t.Fatalf("table[0]: want %f, got %f", table[0], m.table[0])
	}
}

func TestLoadBadMagic(t *testing.T) {
	path := t.TempDir() + "/bad.bin"
	f, _ := os.Create(path)
	var hdr [32]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0xDEADBEEF)
	f.Write(hdr[:])
	f.Close()

	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("/nonexistent/model.bin")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadTruncated(t *testing.T) {
	// Valid header but truncated data — write only the header + partial table.
	path := t.TempDir() + "/truncated.bin"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	bucket, dim, numLabels := 32, 8, 2
	var hdr [32]byte
	binary.LittleEndian.PutUint32(hdr[0:4], modelMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], modelVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(bucket))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(dim))
	binary.LittleEndian.PutUint32(hdr[16:20], 4)
	binary.LittleEndian.PutUint32(hdr[20:24], 16)
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(numLabels))
	f.Write(hdr[:])
	// Write only half the expected table, then truncate.
	partialTable := make([]float32, (bucket*dim)/2)
	binary.Write(f, binary.LittleEndian, partialTable)
	f.Close()

	_, err = Load(path)
	if err == nil {
		t.Fatal("expected error for truncated model")
	}
}

func TestLoadCorruptWeights(t *testing.T) {
	// Full table but no weights — should error on readFloats.
	path := t.TempDir() + "/corrupt.bin"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	bucket, dim, numLabels := 16, 4, 2
	var hdr [32]byte
	binary.LittleEndian.PutUint32(hdr[0:4], modelMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], modelVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(bucket))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(dim))
	binary.LittleEndian.PutUint32(hdr[16:20], 4)
	binary.LittleEndian.PutUint32(hdr[20:24], 16)
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(numLabels))
	f.Write(hdr[:])
	// Write full table.
	fullTable := make([]float32, bucket*dim)
	binary.Write(f, binary.LittleEndian, fullTable)
	// Write partial weights (incomplete — should fail on bias or platt read).
	partialWeights := make([]float32, 1) // only 1 float, need numLabels*dim = 8
	binary.Write(f, binary.LittleEndian, partialWeights)
	f.Close()

	_, err = Load(path)
	if err == nil {
		t.Fatal("expected error for corrupt weights")
	}
}
