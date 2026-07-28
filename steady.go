package steady

import "math"

type workspace struct {
	hidden         []float32
	logits         []float32
	probs          []float32
	kinds          []int
	featureIndices []int
	temporal       []float32
}

func (m *Model) newWorkspace() *workspace {
	outputs := len(m.metadata.Labels)
	if m.metadata.ArtifactFormat == int(modelVersion) {
		outputs = len(m.metadata.Heads)
	}
	return &workspace{
		hidden:         make([]float32, m.dim),
		logits:         make([]float32, outputs),
		probs:          make([]float32, outputs),
		kinds:          make([]int, 0, len(m.metadata.Labels)),
		featureIndices: make([]int, 0, 8192),
		temporal:       make([]float32, m.temporalFeatures),
	}
}

func (m *Model) infer(text string, work *workspace) bool {
	if m == nil || work == nil || text == "" {
		return false
	}
	if m.metadata.ArtifactFormat == int(modelVersion) {
		work.featureIndices = v4Predict(
			text, m.table, m.weights, m.bias, m.temperatures,
			m.bucket, m.dim, m.temporalFeatures, m.minN, m.maxN,
			work.hidden,
			work.probs, work.featureIndices[:0], work.temporal,
		)
		work.kinds = work.kinds[:0]
		shortAccepted := m.v4HeadAccepted(0, work.probs[0], text)
		longAccepted := m.v4HeadAccepted(1, work.probs[1], text)
		if shortAccepted != longAccepted {
			if shortAccepted {
				work.kinds = append(work.kinds, 0)
			} else {
				work.kinds = append(work.kinds, 2)
			}
		} else {
			work.kinds = append(work.kinds, 1)
		}
		return len(work.featureIndices) > 0
	}
	if encodeInto(text, m.table, m.bucket, m.dim, m.minN, m.maxN, work.hidden) == 0 {
		return false
	}
	predictLogits(work.hidden, m.weights, m.bias, work.logits, m.dim)
	softmax(work.logits, work.probs, m.temperature)
	work.kinds = predictionKinds(work.probs, m.metadata.Labels, m.quantiles, work.kinds)
	return true
}

func (m *Model) v4HeadAccepted(head int, probability float32, text string) bool {
	if head < 0 || head >= len(m.thresholds) || head*2+1 >= len(m.quantiles) {
		return false
	}
	positiveIncluded := 1-probability <= m.quantiles[head*2]
	negativeIncluded := probability <= m.quantiles[head*2+1]
	if head == 1 && !longCueEligible(text) {
		return false
	}
	return positiveIncluded && !negativeIncluded && probability >= m.thresholds[head]
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
		Kinds: make([]string, len(work.kinds)),
	}
	if m.metadata.ArtifactFormat == int(modelVersion) {
		medium := max(float32(0), 1-work.probs[0]-work.probs[1])
		total := work.probs[0] + medium + work.probs[1]
		result.Probabilities = []float32{
			work.probs[0] / total, medium / total, work.probs[1] / total,
		}
	} else {
		result.Probabilities = append([]float32(nil), work.probs...)
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
	if m.metadata.ArtifactFormat == int(modelVersion) {
		switch index {
		case 0:
			return "short", work.probs[0], true
		case 2:
			return "long", work.probs[1], true
		default:
			return "", 0, false
		}
	}
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
