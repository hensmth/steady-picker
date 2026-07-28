package steady

import "math"

type workspace struct {
	hidden []float32
	logits []float32
	probs  []float32
	kinds  []int
}

func (m *Model) newWorkspace() *workspace {
	return &workspace{
		hidden: make([]float32, m.dim),
		logits: make([]float32, len(m.metadata.Labels)),
		probs:  make([]float32, len(m.metadata.Labels)),
		kinds:  make([]int, 0, len(m.metadata.Labels)),
	}
}

func (m *Model) infer(text string, work *workspace) bool {
	if m == nil || work == nil || text == "" {
		return false
	}
	if encodeInto(text, m.table, m.bucket, m.dim, m.minN, m.maxN, work.hidden) == 0 {
		return false
	}
	predictLogits(work.hidden, m.weights, m.bias, work.logits, m.dim)
	softmax(work.logits, work.probs, m.temperature)
	work.kinds = predictionKinds(work.probs, m.metadata.Labels, m.quantiles, work.kinds)
	return true
}

// Classify returns an owned prediction set and is safe for concurrent use.
func (m *Model) Classify(text string) PredictionSet {
	if m == nil {
		return PredictionSet{}
	}
	work := m.workspaces.Get().(*workspace)
	defer m.workspaces.Put(work)
	if !m.infer(text, work) {
		return PredictionSet{}
	}
	result := PredictionSet{
		Kinds:         make([]string, len(work.kinds)),
		Probabilities: append([]float32(nil), work.probs...),
	}
	for i, labelIndex := range work.kinds {
		result.Kinds[i] = m.metadata.Labels[labelIndex]
	}
	return result
}

// DebugResult contains caller-owned intermediate inference values.
type DebugResult struct {
	Logits        []float32 `json:"logits"`
	Probabilities []float32 `json:"probabilities"`
	Quantiles     []float32 `json:"quantiles"`
	Thresholds    []float32 `json:"thresholds"`
	Kinds         []string  `json:"kinds"`
	IsEmpty       bool      `json:"is_empty"`
}

// ClassifyDebug is safe for concurrent use.
func (m *Model) ClassifyDebug(text string) DebugResult {
	if m == nil {
		return DebugResult{IsEmpty: true}
	}
	work := m.workspaces.Get().(*workspace)
	defer m.workspaces.Put(work)
	if !m.infer(text, work) {
		return DebugResult{IsEmpty: true}
	}
	result := DebugResult{
		Logits:        append([]float32(nil), work.logits...),
		Probabilities: append([]float32(nil), work.probs...),
		Quantiles:     append([]float32(nil), m.quantiles...),
		Thresholds:    append([]float32(nil), m.thresholds...),
		Kinds:         make([]string, len(work.kinds)),
	}
	for i, labelIndex := range work.kinds {
		result.Kinds[i] = m.metadata.Labels[labelIndex]
	}
	return result
}

func (m *Model) pickDecision(text string) (string, float32, bool) {
	if m == nil {
		return "", 0, false
	}
	work := m.workspaces.Get().(*workspace)
	defer m.workspaces.Put(work)
	if !m.infer(text, work) || len(work.kinds) != 1 {
		return "", 0, false
	}
	index := work.kinds[0]
	label := m.metadata.Labels[index]
	if (label != "short" && label != "long") || work.probs[index] < m.thresholds[index] {
		return "", 0, false
	}
	return label, work.probs[index], true
}

func bestProbability(probabilities []float32) (int, float32) {
	index := -1
	best := float32(-math.MaxFloat32)
	for i, probability := range probabilities {
		if probability > best {
			index, best = i, probability
		}
	}
	return index, best
}
