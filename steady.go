package steady

import "github.com/xDarkicex/memory"

// SetLabelNames sets the human-readable label names used in prediction sets.
func (m *Model) SetLabelNames(names []string) {
	if m == nil || len(names) != m.numLabels {
		return
	}
	m.labelNames = names
}

// Classify runs the full classification pipeline on text. It returns a
// calibrated prediction set with conformal coverage guarantees. An empty set
// indicates that the text could not be classified with sufficient confidence
// (noise or out-of-distribution).
func (m *Model) Classify(text string) PredictionSet {
	if m == nil || m.scratchPool == nil {
		return PredictionSet{}
	}
	m.scratchPool.Reset()

	hidden := Encode([]byte(text), m.table, m.bucket, m.dim, m.scratchPool)
	if len(hidden) == 0 {
		return PredictionSet{}
	}
	logits := memoryMustPoolSlice[float32](m.scratchPool, m.numLabels)
	logits = logits[:m.numLabels]
	PredictLogits(hidden, m.weights, m.bias, logits)

	calibrated := memoryMustPoolSlice[float32](m.scratchPool, m.numLabels)
	calibrated = calibrated[:m.numLabels]
	for i := range m.numLabels {
		calibrated[i] = ApplyPlatt(
			decisionScore(logits[i]),
			m.plattA[i],
			m.plattB[i],
		)
	}
	return PredictSet(calibrated, m.labelNames, m.q, m.scratchPool)
}

// DebugResult holds raw intermediate values for debugging a classification.
type DebugResult struct {
	Logits     []float32
	Calibrated []float32
	PlattA     []float32
	PlattB     []float32
	Q          float32
	IsEmpty    bool
	Kinds      []string
}

// ClassifyDebug runs classification and returns raw intermediate values.
func (m *Model) ClassifyDebug(text string) DebugResult {
	if m == nil || m.scratchPool == nil {
		return DebugResult{IsEmpty: true}
	}
	m.scratchPool.Reset()

	hidden := Encode([]byte(text), m.table, m.bucket, m.dim, m.scratchPool)
	if len(hidden) == 0 {
		return DebugResult{IsEmpty: true}
	}
	logits := memoryMustPoolSlice[float32](m.scratchPool, m.numLabels)
	logits = logits[:m.numLabels]
	PredictLogits(hidden, m.weights, m.bias, logits)

	calibrated := memoryMustPoolSlice[float32](m.scratchPool, m.numLabels)
	calibrated = calibrated[:m.numLabels]
	for i := range m.numLabels {
		calibrated[i] = ApplyPlatt(
			decisionScore(logits[i]),
			m.plattA[i],
			m.plattB[i],
		)
	}
	ps := PredictSet(calibrated, m.labelNames, m.q, m.scratchPool)

	n := m.numLabels
	lcopy := memoryMustPoolSlice[float32](m.scratchPool, n)
	lcopy = lcopy[:n]
	copy(lcopy, logits)
	ccopy := memoryMustPoolSlice[float32](m.scratchPool, n)
	ccopy = ccopy[:n]
	copy(ccopy, calibrated)
	acopy := memoryMustPoolSlice[float32](m.scratchPool, n)
	acopy = acopy[:n]
	copy(acopy, m.plattA)
	bcopy := memoryMustPoolSlice[float32](m.scratchPool, n)
	bcopy = bcopy[:n]
	copy(bcopy, m.plattB)

	return DebugResult{
		Logits: lcopy, Calibrated: ccopy, PlattA: acopy, PlattB: bcopy,
		Q: m.q, IsEmpty: ps.IsEmpty(), Kinds: ps.Kinds,
	}
}

// memoryMustPoolSlice is an internal helper that allocates a typed slice from
// the pool. Panics if allocation fails.
func memoryMustPoolSlice[T any](pool *memory.Pool, cap int) []T {
	s := memory.MustPoolSlice[T](pool, cap)
	return s
}
