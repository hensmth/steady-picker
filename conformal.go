package steady

import (
	"sort"

	"github.com/xDarkicex/memory"
)

// PredictionSet is the output of a conformal classification. It contains the
// set of labels that meet the coverage guarantee at the configured significance
// level. An empty set indicates out-of-distribution or noise input.
type PredictionSet struct {
	Kinds       []string
	Confidences []float32
}

// IsEmpty returns true if the prediction set contains no labels.
func (ps PredictionSet) IsEmpty() bool {
	return len(ps.Kinds) == 0
}

// ComputeQuantile computes the conformal quantile Q from calibration scores.
// scores[i] is the non-conformity score (1 - probability for the true label)
// for calibration example i. alpha is the desired error rate (e.g., 0.05 for 95%).
func ComputeQuantile(scores []float32, alpha float64) float32 {
	if len(scores) == 0 {
		return 1.0
	}
	sorted := make([]float32, len(scores))
	copy(sorted, scores)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := float64(len(sorted))
	idx := int((n + 1) * (1.0 - alpha))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	if idx < 0 {
		idx = 0
	}
	return sorted[idx]
}

// PredictSet returns the prediction set for calibrated probabilities.
// For each label i, include if (1.0 - probs[i]) <= q.
// If no labels qualify, returns an empty set. Uses pool for output slices.
func PredictSet(probs []float32, labelNames []string, q float32, pool *memory.Pool) PredictionSet {
	ks := memory.MustPoolSlice[string](pool, len(probs))
	cs := memory.MustPoolSlice[float32](pool, len(probs))
	for i, p := range probs {
		if (1.0 - p) <= q {
			ks = append(ks, labelNames[i])
			cs = append(cs, p)
		}
	}
	return PredictionSet{Kinds: ks, Confidences: cs}
}
