package steady

import (
	"math"

	"github.com/xDarkicex/memory"
)

// ApplyPlatt returns the Platt-calibrated probability for a raw score.
// P(calibrated) = 1 / (1 + exp(-(A * score + B)))
func ApplyPlatt(score, a, b float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-(a*score + b)))))
}

// CalibratePlatt fits Platt scaling parameters (A, B) for each label using
// Newton's method on a calibration set. rawScores[i] is the vector of
// uncalibrated scores for example i, labels[i] is the true label index.
// Returns two slices of length numLabels allocated from pool.
func CalibratePlatt(pool *memory.Pool, rawScores [][]float32, labels []int, numLabels int) ([]float32, []float32) {
	n := len(rawScores)
	if n == 0 || numLabels == 0 {
		return nil, nil
	}
	a := memory.MustPoolSlice[float32](pool, numLabels)
	a = a[:numLabels]
	b := memory.MustPoolSlice[float32](pool, numLabels)
	b = b[:numLabels]
	for label := range numLabels {
		a[label], b[label] = fitPlattLabel(rawScores, labels, label)
	}
	return a, b
}

// fitPlattLabel fits (A, B) for a single label using Newton's method.
func fitPlattLabel(rawScores [][]float32, labels []int, targetLabel int) (float32, float32) {
	// Prior: A=0, B=0 (identity mapping)
	var a, b float32 = 0, 0
	const (
		lr       float32 = 1.0
		iter     int     = 2000
		minDelta float32 = 1e-5
	)
	for range iter {
		var dA, dB float32
		for i, scores := range rawScores {
			yi := float32(1.0)
			if labels[i] != targetLabel {
				yi = 0.0
			}
			f := float32(math.Exp(float64(a*scores[targetLabel] + b)))
			// Platt objective gradient: (yi - 1/(1+exp(-f))) * d/dA and d/dB
			p := 1.0 / (1.0 + 1.0/f)
			diff := yi - p
			dA += diff * scores[targetLabel]
			dB += diff
		}
		dA /= float32(len(rawScores))
		dB /= float32(len(rawScores))
		if absF32(dA) < minDelta && absF32(dB) < minDelta {
			break
		}
		a += lr * dA
		b += lr * dB
	}
	return a, b
}

func absF32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
