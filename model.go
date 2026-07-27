package steady

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	// modelMagic is the 4-byte identifier for steady model files ("BYTE").
	modelMagic   uint32 = 0x42595445
	modelVersion uint32 = 2
	headerSize   int64  = 32
)

// Model holds a loaded classification model. All fields are read-only after Load.
type Model struct {
	table       []float32 // mmap'd embedding table, bucket × dim
	weights     []float32 // OVA weights, numLabels × dim
	bias        []float32 // OVA bias, numLabels
	plattA      []float32 // Platt slope, numLabels
	plattB      []float32 // Platt intercept, numLabels
	q           float32   // conformal quantile
	bucket      int
	dim         int
	numLabels   int
	labelNames  []string
	modelPool   *memory.Pool
	scratchPool *memory.Pool
	tableMapped bool
}

// Load opens a model file and returns the loaded Model. The caller must call
// Close to release resources.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("steady: open model: %w", err)
	}
	defer f.Close()

	var hdr [headerSize]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return nil, fmt.Errorf("steady: read header: %w", err)
	}
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("steady: stat model: %w", err)
	}
	bucket, dim, numLabels, err := validateModelHeader(hdr[:], stat.Size())
	if err != nil {
		return nil, err
	}

	// Map the embedding table.
	tableBytes := bucket * dim * 4
	tableRaw, err := memory.MmapFileReadOnly(int(f.Fd()), headerSize, tableBytes)
	if err != nil {
		return nil, fmt.Errorf("steady: mmap table: %w", err)
	}
	table := unsafe.Slice((*float32)(unsafe.Pointer(&tableRaw[0])), bucket*dim)

	modelPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  16 * 1024 * 1024,
		SlabSize:  1024 * 1024,
		SlabCount: 8,
	}, 64)
	if err != nil {
		memory.Munmap(tableRaw)
		return nil, fmt.Errorf("steady: create model pool: %w", err)
	}

	scratchPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  2 * 1024 * 1024,
		SlabSize:  256 * 1024,
		SlabCount: 4,
	}, 64)
	if err != nil {
		modelPool.Free()
		memory.Munmap(tableRaw)
		return nil, fmt.Errorf("steady: create scratch pool: %w", err)
	}

	// Seek past the embedding table (already mmap'd).
	if _, err := f.Seek(headerSize+int64(tableBytes), 0); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: seek past table: %w", err)
	}

	// Read the small arrays from the file (after the table).
	weights := memory.MustPoolSlice[float32](modelPool, numLabels*dim)
	weights = weights[:numLabels*dim]
	if err := readFloats(f, weights); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read weights: %w", err)
	}

	bias := memory.MustPoolSlice[float32](modelPool, numLabels)
	bias = bias[:numLabels]
	if err := readFloats(f, bias); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read bias: %w", err)
	}

	plattA := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattA = plattA[:numLabels]
	if err := readFloats(f, plattA); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read plattA: %w", err)
	}

	plattB := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattB = plattB[:numLabels]
	if err := readFloats(f, plattB); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read plattB: %w", err)
	}

	var qRaw [4]byte
	if _, err := f.Read(qRaw[:]); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read quantile: %w", err)
	}
	q := float32fromle(qRaw)

	m := &Model{
		table:       table,
		weights:     weights,
		bias:        bias,
		plattA:      plattA,
		plattB:      plattB,
		q:           q,
		bucket:      bucket,
		dim:         dim,
		numLabels:   numLabels,
		modelPool:   modelPool,
		scratchPool: scratchPool,
		tableMapped: true,
	}
	m.setDefaultLabelNames()
	return m, nil
}

// LoadBytes loads a model from an in-memory artifact. Unlike Load, it copies
// the embedding table instead of memory-mapping it. The caller must call Close.
func LoadBytes(data []byte) (*Model, error) {
	if len(data) < int(headerSize) {
		return nil, errors.New("steady: model is smaller than its header")
	}
	bucket, dim, numLabels, err := validateModelHeader(data[:headerSize], int64(len(data)))
	if err != nil {
		return nil, err
	}

	modelPool, scratchPool, err := newModelPools()
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*Model, error) {
		modelPool.Free()
		scratchPool.Free()
		return nil, err
	}

	reader := bytes.NewReader(data[headerSize:])
	table := make([]float32, bucket*dim)
	if err := readFloats(reader, table); err != nil {
		return fail(fmt.Errorf("steady: read table: %w", err))
	}
	weights := memory.MustPoolSlice[float32](modelPool, numLabels*dim)
	weights = weights[:numLabels*dim]
	if err := readFloats(reader, weights); err != nil {
		return fail(fmt.Errorf("steady: read weights: %w", err))
	}
	bias := memory.MustPoolSlice[float32](modelPool, numLabels)
	bias = bias[:numLabels]
	if err := readFloats(reader, bias); err != nil {
		return fail(fmt.Errorf("steady: read bias: %w", err))
	}
	plattA := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattA = plattA[:numLabels]
	if err := readFloats(reader, plattA); err != nil {
		return fail(fmt.Errorf("steady: read plattA: %w", err))
	}
	plattB := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattB = plattB[:numLabels]
	if err := readFloats(reader, plattB); err != nil {
		return fail(fmt.Errorf("steady: read plattB: %w", err))
	}
	var qRaw [4]byte
	if _, err := io.ReadFull(reader, qRaw[:]); err != nil {
		return fail(fmt.Errorf("steady: read quantile: %w", err))
	}

	model := &Model{
		table: table, weights: weights, bias: bias, plattA: plattA, plattB: plattB,
		q: float32fromle(qRaw), bucket: bucket, dim: dim, numLabels: numLabels,
		modelPool: modelPool, scratchPool: scratchPool,
	}
	model.setDefaultLabelNames()
	return model, nil
}

// Close releases resources held by the model.
func (m *Model) Close() error {
	if m.table != nil {
		if m.tableMapped {
			tableRaw := unsafe.Slice((*byte)(unsafe.Pointer(&m.table[0])), len(m.table)*4)
			if err := memory.Munmap(tableRaw); err != nil {
				return err
			}
		}
		m.table = nil
	}
	if m.modelPool != nil {
		m.modelPool.Free()
		m.modelPool = nil
	}
	if m.scratchPool != nil {
		m.scratchPool.Free()
		m.scratchPool = nil
	}
	return nil
}

func readFloats(reader io.Reader, dst []float32) error {
	return binary.Read(reader, binary.LittleEndian, dst)
}

func float32fromle(b [4]byte) float32 {
	return float32frombits(binary.LittleEndian.Uint32(b[:]))
}

func float32frombits(u uint32) float32 {
	return *(*float32)(unsafe.Pointer(&u))
}

// setDefaultLabelNames populates labelNames with generic "label_0".."label_N-1"
// defaults. Callers should use SetLabelNames to override with meaningful names.
func (m *Model) setDefaultLabelNames() {
	names := make([]string, m.numLabels)
	for i := range names {
		names[i] = fmt.Sprintf("label_%d", i)
	}
	m.labelNames = names
}

func validateModelHeader(hdr []byte, actualSize int64) (int, int, int, error) {
	if len(hdr) < int(headerSize) {
		return 0, 0, 0, errors.New("steady: model is smaller than its header")
	}
	if binary.LittleEndian.Uint32(hdr[0:4]) != modelMagic {
		return 0, 0, 0, errors.New("steady: not a steady model file")
	}
	version := binary.LittleEndian.Uint32(hdr[4:8])
	if version != modelVersion {
		return 0, 0, 0, fmt.Errorf("steady: unsupported model version %d", version)
	}
	bucket := int(binary.LittleEndian.Uint32(hdr[8:12]))
	dim := int(binary.LittleEndian.Uint32(hdr[12:16]))
	numLabels := int(binary.LittleEndian.Uint32(hdr[24:28]))
	if bucket <= 0 || dim <= 0 || numLabels <= 0 {
		return 0, 0, 0, errors.New("steady: invalid header dimensions")
	}
	expectedSize := headerSize +
		int64(bucket)*int64(dim)*4 +
		int64(numLabels)*int64(dim)*4 +
		int64(numLabels)*4*3 + 4
	if actualSize != expectedSize {
		return 0, 0, 0, fmt.Errorf(
			"steady: invalid model size: got %d, want %d",
			actualSize, expectedSize,
		)
	}
	return bucket, dim, numLabels, nil
}

func newModelPools() (*memory.Pool, *memory.Pool, error) {
	modelPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  16 * 1024 * 1024,
		SlabSize:  1024 * 1024,
		SlabCount: 8,
	}, 64)
	if err != nil {
		return nil, nil, fmt.Errorf("steady: create model pool: %w", err)
	}
	scratchPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  2 * 1024 * 1024,
		SlabSize:  256 * 1024,
		SlabCount: 4,
	}, 64)
	if err != nil {
		modelPool.Free()
		return nil, nil, fmt.Errorf("steady: create scratch pool: %w", err)
	}
	return modelPool, scratchPool, nil
}
