package steady

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/xDarkicex/memory"
)

// TrainConfig holds hyperparameters for training a classification model.
type TrainConfig struct {
	Input         string
	Output        string
	Bucket        int
	Dim           int
	Epochs        int
	LR            float32
	Alpha         float64
	LabelNames    []string
	NumGoroutines int
	CalibSplit    float64
	Seed          uint64
}

// DefaultTrainConfig returns a TrainConfig with sensible defaults.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Bucket:        20000,
		Dim:           64,
		Epochs:        25,
		LR:            0.1,
		Alpha:         0.05,
		NumGoroutines: 1,
		CalibSplit:    0.15,
		Seed:          42,
		LabelNames:    []string{"label_0", "label_1", "label_2", "label_3", "label_4", "label_5"},
	}
}

type example struct {
	text  string
	label int
}

// Train runs the full training pipeline and writes the model artifact.
func Train(cfg TrainConfig) error {
	if cfg.Input == "" || cfg.Output == "" {
		return fmt.Errorf("steady: input and output paths are required")
	}
	if cfg.NumGoroutines <= 0 {
		cfg.NumGoroutines = 1
	}
	if len(cfg.LabelNames) < 2 {
		return fmt.Errorf("steady: at least two labels are required")
	}
	if cfg.CalibSplit <= 0 || cfg.CalibSplit >= 0.5 {
		return fmt.Errorf("steady: calibration split must be greater than 0 and less than 0.5")
	}
	seedRand(cfg.Seed)

	examples, err := parseExamples(cfg.Input, cfg.LabelNames)
	if err != nil {
		return fmt.Errorf("steady: parse input: %w", err)
	}
	if len(examples) == 0 {
		return fmt.Errorf("steady: no training examples found")
	}
	shuffle(examples)
	trainExamples, calibExamples, err := splitCalibration(
		examples,
		len(cfg.LabelNames),
		cfg.CalibSplit,
	)
	if err != nil {
		return err
	}

	table, err := initTable(cfg.Bucket, cfg.Dim)
	if err != nil {
		return err
	}
	defer memory.Munmap(unsafeSliceBytes(table))
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: 64 * 1024 * 1024, SlabSize: 1024 * 1024, SlabCount: 16,
	}, 64)
	if err != nil {
		return err
	}
	defer pool.Free()

	numLabels := len(cfg.LabelNames)
	weights := memory.MustPoolSlice[float32](pool, numLabels*cfg.Dim)
	weights = weights[:numLabels*cfg.Dim]
	scale := float32(1.0 / float64(cfg.Dim))
	rngW := uint32(54321)
	for i := range weights {
		rngW ^= rngW << 13
		rngW ^= rngW >> 17
		rngW ^= rngW << 5
		weights[i] = (float32(rngW)/float32(0xffffffff)*2 - 1) * scale
	}

	bias := memory.MustPoolSlice[float32](pool, numLabels)
	bias = bias[:numLabels]

	err = trainEmbeddings(trainExamples, table, weights, bias, cfg)
	if err != nil {
		return err
	}

	plattA, plattB, calibScores := calibratePlatt(pool, calibExamples, table, weights, bias, cfg)
	q := ComputeQuantile(calibScores, cfg.Alpha)

	return writeModel(cfg.Output, table, weights, bias, plattA, plattB, q, cfg)
}

func splitCalibration(
	examples []example,
	numLabels int,
	fraction float64,
) ([]example, []example, error) {
	counts := make([]int, numLabels)
	for _, ex := range examples {
		counts[ex.label]++
	}
	targets := make([]int, numLabels)
	for label, count := range counts {
		if count < 2 {
			return nil, nil, fmt.Errorf(
				"steady: label %d requires at least two examples", label,
			)
		}
		targets[label] = maxInt(1, int(float64(count)*fraction))
		if targets[label] >= count {
			targets[label] = count - 1
		}
	}
	selected := make([]int, numLabels)
	train := make([]example, 0, len(examples))
	calib := make([]example, 0, int(float64(len(examples))*fraction))
	for _, ex := range examples {
		if selected[ex.label] < targets[ex.label] {
			calib = append(calib, ex)
			selected[ex.label]++
		} else {
			train = append(train, ex)
		}
	}
	return train, calib, nil
}

func parseExamples(path string, labelNames []string) ([]example, error) {
	labelIdx := make(map[string]int, len(labelNames))
	for i, name := range labelNames {
		labelIdx[name] = i
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var examples []example
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		const prefix = "__label__"
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		rest := line[len(prefix):]
		space := strings.IndexByte(rest, ' ')
		if space < 0 {
			continue
		}
		name := rest[:space]
		text := strings.TrimSpace(rest[space+1:])
		if text == "" {
			continue
		}
		idx, ok := labelIdx[name]
		if !ok {
			continue
		}
		examples = append(examples, example{text: text, label: idx})
	}
	return examples, sc.Err()
}

func initTable(bucket, dim int) ([]float32, error) {
	f, err := os.CreateTemp("", "steady_train_*.bin")
	if err != nil {
		return nil, err
	}
	size := int64(bucket * dim * 4)
	if err := f.Truncate(size); err != nil {
		f.Close()
		return nil, err
	}
	name := f.Name()
	f.Close()
	fd, err := os.OpenFile(name, os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	raw, err := memory.MmapFile(int(fd.Fd()), 0, int(size), true)
	if err != nil {
		os.Remove(name)
		return nil, err
	}
	os.Remove(name) // unlink now; mapping survives until Munmap
	table := unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), bucket*dim)
	scale := float32(1.0 / float64(dim))
	rng := uint32(12345)
	for i := range table {
		rng ^= rng << 13
		rng ^= rng >> 17
		rng ^= rng << 5
		table[i] = (float32(rng)/float32(0xffffffff)*2 - 1) * scale
	}
	return table, nil
}

func trainEmbeddings(examples []example, table, weights, bias []float32, cfg TrainConfig) error {

	numWorkers := cfg.NumGoroutines
	shardSize := (len(examples) + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * shardSize
		end := start + shardSize
		if end > len(examples) {
			end = len(examples)
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(shard []example) {
			defer wg.Done()
			hogwildWorker(shard, table, weights, bias, cfg)
		}(examples[start:end])
	}
	wg.Wait()

	return nil
}

// hogwildWorker runs Hogwild! SGD jointly on embeddings and logistic head.
func hogwildWorker(shard []example, table, weights, bias []float32, cfg TrainConfig) {
	pool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize: 16 * 1024 * 1024, SlabSize: 1024 * 1024, SlabCount: 8,
	}, 64)
	if err != nil {
		return
	}
	defer pool.Free()
	numLabels := len(cfg.LabelNames)

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		lr := cfg.LR * (1.0 - float32(epoch)/float32(cfg.Epochs))
		for _, ex := range shard {
			pool.Reset()
			hidden := Encode([]byte(ex.text), table, cfg.Bucket, cfg.Dim, pool)

			logits := memory.MustPoolSlice[float32](pool, numLabels)
			logits = logits[:numLabels]
			PredictLogits(hidden, weights, bias, logits)

			gradHidden := memory.MustPoolSlice[float32](pool, cfg.Dim)
			gradHidden = gradHidden[:cfg.Dim]
			for j := range gradHidden {
				gradHidden[j] = 0
			}

			// The preset labels are mutually exclusive and few in number.
			// Update the positive and every negative label so OVA probabilities
			// are trained and calibrated against the same complete objective.
			for l := 0; l < numLabels; l++ {
				target := float32(0.0)
				if l == ex.label {
					target = 1.0
				}
				errVal := (logits[l] - target) * lr

				// accumulate gradient for hidden layer
				for j := 0; j < cfg.Dim; j++ {
					gradHidden[j] += errVal * weights[l*cfg.Dim+j]
				}

				// Update bias and weights with simple SGD
				for j := 0; j < cfg.Dim; j++ {
					atomicAddFloat32(&weights[l*cfg.Dim+j], -errVal*hidden[j]-1e-4*weights[l*cfg.Dim+j])
				}
				if target == 1.0 {
					atomicAddFloat32(&bias[l], -errVal-1e-4*bias[l])
				} else {
					atomicAddFloat32(&bias[l], -errVal-1e-4*bias[l])
				}
			}

			// Center gradHidden to prevent embedding drift!
			var meanGrad float32
			for j := 0; j < cfg.Dim; j++ {
				meanGrad += gradHidden[j]
			}
			meanGrad /= float32(cfg.Dim)
			for j := 0; j < cfg.Dim; j++ {
				gradHidden[j] -= meanGrad
			}

			// Clip gradHidden to prevent exploding gradients
			var norm float32
			for j := 0; j < cfg.Dim; j++ {
				norm += gradHidden[j] * gradHidden[j]
			}
			norm = float32(math.Sqrt(float64(norm)))
			if norm > 1.0 {
				invNorm := 1.0 / norm
				for j := 0; j < cfg.Dim; j++ {
					gradHidden[j] *= invNorm
				}
			}

			// Update embeddings
			updateEmbeddings([]byte(ex.text), table, cfg.Bucket, cfg.Dim, gradHidden, pool)
		}
	}
}

// updateEmbeddings applies gradient descent with weight decay to embedding rows.
func updateEmbeddings(text []byte, table []float32, bucket, dim int, gradHidden []float32, pool *memory.Pool) {
	// Worst case n-grams: len(text) * len(windows)
	maxNGrams := len(text) * len(windows)
	if maxNGrams == 0 {
		return
	}

	hashes := memory.MustPoolSlice[uint32](pool, maxNGrams)
	hashes = hashes[:maxNGrams]
	var count int

	for _, n := range windows {
		if n > len(text) {
			continue
		}
		for i := 0; i <= len(text)-n; i++ {
			h := hashWindow(text[i : i+n])
			hashes[count] = h % uint32(bucket)
			count++
		}
	}
	if count == 0 {
		return
	}

	// Sort to bring duplicates together
	slices.Sort(hashes[:count])

	// Encode averages all n-gram rows. Apply each row's frequency divided by
	// the total count so this update is the actual derivative of that average.
	for i := 0; i < count; {
		j := i + 1
		for j < count && hashes[j] == hashes[i] {
			j++
		}
		frequencyScale := float32(j-i) / float32(count)
		base := int(hashes[i]) * dim
		for d := 0; d < dim; d++ {
			const tDecay = float32(1e-4)
			atomicAddFloat32(
				&table[base+d],
				-gradHidden[d]*frequencyScale-tDecay*table[base+d],
			)
		}
		i = j
	}
}

// calibratePlatt returns pool-backed slices since we are passing a pool
func calibratePlatt(pool *memory.Pool, examples []example, table []float32, weights, bias []float32, cfg TrainConfig) ([]float32, []float32, []float32) {
	numLabels := len(cfg.LabelNames)
	n := len(examples)

	rawScores := memory.MustPoolSlice[[]float32](pool, n)
	rawScores = rawScores[:n]
	labels := memory.MustPoolSlice[int](pool, n)
	labels = labels[:n]
	for i, ex := range examples {
		hidden := Encode([]byte(ex.text), table, cfg.Bucket, cfg.Dim, pool)
		logits := memory.MustPoolSlice[float32](pool, numLabels)
		logits = logits[:numLabels]
		PredictLogits(hidden, weights, bias, logits)
		s := memory.MustPoolSlice[float32](pool, numLabels)
		s = s[:numLabels]
		for label, probability := range logits {
			s[label] = decisionScore(probability)
		}
		rawScores[i] = s
		labels[i] = ex.label
	}
	plattA, plattB := CalibratePlatt(pool, rawScores, labels, numLabels)
	cScores := memory.MustPoolSlice[float32](pool, n)
	cScores = cScores[:n]
	for i, ex := range examples {
		p := ApplyPlatt(rawScores[i][ex.label], plattA[ex.label], plattB[ex.label])
		cScores[i] = 1.0 - p
	}
	return plattA, plattB, cScores
}

func writeModel(path string, table, weights, bias, plattA, plattB []float32, q float32, cfg TrainConfig) error {
	directory := filepath.Dir(path)
	f, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	temporary := f.Name()
	defer os.Remove(temporary)
	if err := f.Chmod(0600); err != nil {
		f.Close()
		return err
	}
	var hdr [32]byte
	binary.LittleEndian.PutUint32(hdr[0:4], modelMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], modelVersion)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(cfg.Bucket))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(cfg.Dim))
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(windows[0]))
	binary.LittleEndian.PutUint32(hdr[20:24], uint32(windows[len(windows)-1]))
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(len(cfg.LabelNames)))
	if _, err := f.Write(hdr[:]); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, table); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, weights); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, bias); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, plattA); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, plattB); err != nil {
		f.Close()
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, q); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func shuffle(examples []example) {
	for i := len(examples) - 1; i > 0; i-- {
		j := int(fastRand() % uint64(i+1))
		examples[i], examples[j] = examples[j], examples[i]
	}
}

var randState atomic.Uint64

func seedRand(seed uint64) { randState.Store(seed) }

func fastRand() uint64 {
	for {
		old := randState.Load()
		next := old*1103515245 + 12345
		if randState.CompareAndSwap(old, next) {
			return next
		}
	}
}

func atomicAddFloat32(addr *float32, delta float32) {
	for {
		old := atomic.LoadUint32((*uint32)(unsafe.Pointer(addr)))
		new := math.Float32bits(math.Float32frombits(old) + delta)
		if atomic.CompareAndSwapUint32((*uint32)(unsafe.Pointer(addr)), old, new) {
			return
		}
	}
}

func unsafeSliceBytes(s []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
