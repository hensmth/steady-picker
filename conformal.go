package steady

import (
	"cmp"
	"math"
	"slices"
)

// PredictionSet is an owned, immutable prediction result.
type PredictionSet struct {
	Kinds         []string  `json:"kinds"`
	Probabilities []float32 `json:"probabilities"`
}

func (p PredictionSet) IsEmpty() bool { return len(p.Kinds) == 0 }

func predictionKinds(probabilities []float32, labels []string, quantiles []float32, dst []int) []int {
	dst = dst[:0]
	for i, probability := range probabilities {
		if i < len(quantiles) && 1-probability <= quantiles[i] {
			dst = append(dst, i)
		}
	}
	return dst
}

func conformalQuantile(scores []float32, alpha float64) float32 {
	if len(scores) == 0 {
		return 1
	}
	ordered := append([]float32(nil), scores...)
	slices.SortFunc(ordered, func(a, b float32) int { return cmp.Compare(a, b) })
	rank := int(math.Ceil(float64(len(ordered)+1)*(1-alpha))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(ordered) {
		rank = len(ordered) - 1
	}
	return ordered[rank]
}
