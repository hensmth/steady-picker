package steady

import "math"

func predictLogits(hidden, weights, bias, out []float32, dim int) {
	for label := range bias {
		sum := bias[label]
		offset := label * dim
		for d := 0; d < dim; d++ {
			sum += hidden[d] * weights[offset+d]
		}
		out[label] = sum
	}
}

func softmax(logits, out []float32, temperature float32) {
	if len(logits) == 0 {
		return
	}
	if temperature <= 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) {
		temperature = 1
	}
	maxValue := float64(logits[0] / temperature)
	for _, value := range logits[1:] {
		scaled := float64(value / temperature)
		if scaled > maxValue {
			maxValue = scaled
		}
	}
	total := 0.0
	for i, value := range logits {
		probability := math.Exp(float64(value/temperature) - maxValue)
		out[i] = float32(probability)
		total += probability
	}
	if total == 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		uniform := float32(1 / float64(len(out)))
		for i := range out {
			out[i] = uniform
		}
		return
	}
	inverse := float32(1 / total)
	for i := range out {
		out[i] *= inverse
	}
}
