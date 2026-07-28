package steady

import (
	"math"
)

const temporalFeatureCount = 16

func sigmoid(value, temperature float32) float32 {
	if temperature <= 0 {
		temperature = 1
	}
	x := float64(value / temperature)
	if x >= 0 {
		z := math.Exp(-x)
		return float32(1 / (1 + z))
	}
	z := math.Exp(x)
	return float32(z / (1 + z))
}

func sparseBuckets(text string, bucket, minN, maxN int, dst []int) []int {
	dst = ngramBuckets(text, bucket, minN, maxN, dst)
	var previous uint64
	havePrevious := false
	for start := 0; start < len(text); {
		for start < len(text) && !asciiWord(text[start]) {
			start++
		}
		if start >= len(text) {
			break
		}
		hash := fnvOffset64 ^ 0x776f7264
		end := start
		for end < len(text) && asciiWord(text[end]) {
			value := text[end]
			if value >= 'A' && value <= 'Z' {
				value += 'a' - 'A'
			}
			hash ^= uint64(value)
			hash *= fnvPrime64
			end++
		}
		dst = append(dst, int(hash%uint64(bucket)))
		if havePrevious {
			pair := previous
			pair ^= hash + 0x9e3779b97f4a7c15 + (pair << 6) + (pair >> 2)
			dst = append(dst, int(pair%uint64(bucket)))
		}
		previous, havePrevious, start = hash, true, end
	}
	return dst
}

func asciiWord(value byte) bool {
	return value >= 'a' && value <= 'z' ||
		value >= 'A' && value <= 'Z' ||
		value >= '0' && value <= '9'
}

func temporalValues(text string, out []float32) {
	clear(out)
	if len(out) < temporalFeatureCount {
		return
	}
	groups := [...][]string{
		{" then ", " next ", " finally "},
		{" before ", " after "},
		{" first ", " second ", " third "},
		{" gradually", " eventually", "progress"},
		{"transform", "turns into", "turn into"},
		{"evolv", "grow", "develop"},
		{"time-lapse", "timelapse", "journey"},
		{"build", "construct", "assemble"},
		{"mix", "add ", "pour", "cook"},
		{"open", "pick", "place", "put "},
		{" and then ", " followed by "},
		{" while ", " meanwhile ", " simultaneously "},
		{"multiple ", "various ", "different "},
		{"from ", " into ", " to "},
		{"crash", "explod", "destroy", "repair"},
		{"walk", "run", "drive", "travel"},
	}
	for index, patterns := range groups {
		count := 0
		for _, pattern := range patterns {
			count += countFold(text, pattern)
		}
		if count > 4 {
			count = 4
		}
		out[index] = float32(count) / 4
	}
}

func countFold(text, pattern string) int {
	if len(pattern) == 0 || len(pattern) > len(text) {
		return 0
	}
	count := 0
	for start := 0; start+len(pattern) <= len(text); start++ {
		match := true
		for offset := range len(pattern) {
			a, b := text[start+offset], pattern[offset]
			if a >= 'A' && a <= 'Z' {
				a += 'a' - 'A'
			}
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			if a != b {
				match = false
				break
			}
		}
		if match {
			count++
			start += len(pattern) - 1
		}
	}
	return count
}

func longCueEligible(text string) bool {
	strong := 0
	for _, pattern := range [...]string{
		" then ", " and then ", " followed by ", " finally ",
		"transform", "turns into", "turn into", "evolv",
		"time-lapse", "timelapse", "gradually", "eventually",
	} {
		strong += countFold(text, pattern)
	}
	if strong > 0 {
		return true
	}
	actions := 0
	for _, pattern := range [...]string{
		" walking", " running", " driving", " traveling", " flying", " taking",
		" opening", " picking", " placing", " putting", " pouring", " adding",
		" mixing", " cutting", " folding", " cooking", " building",
		" constructing", " assembling", " growing", " blooming", " sprouting",
		" crashing", " exploding", " destroying", " repairing", " shooting",
		" firing", " fighting", " attacking", " drinking", " eating",
	} {
		if countFold(text, pattern) > 0 {
			actions++
		}
	}
	return actions >= 2
}

func v4Predict(
	text string,
	table, weights, bias, temperatures []float32,
	bucket, dimension, temporal, minN, maxN int,
	hidden []float32,
	out []float32,
	indices []int,
	temporalBuffer []float32,
) []int {
	indices = sparseBuckets(text, bucket, minN, maxN, indices)
	temporalValues(text, temporalBuffer)
	clear(hidden)
	scale := float32(0)
	if len(indices) > 0 {
		scale = float32(1 / float64(len(indices)))
		for _, index := range indices {
			base := index * dimension
			for feature := 0; feature < dimension; feature++ {
				hidden[feature] += table[base+feature]
			}
		}
		for feature := range dimension {
			hidden[feature] *= scale
		}
	}
	stride := dimension + temporal
	for head := range bias {
		logit := bias[head]
		base := head * stride
		for feature := 0; feature < dimension; feature++ {
			logit += weights[base+feature] * hidden[feature]
		}
		for feature := 0; feature < temporal; feature++ {
			logit += weights[base+dimension+feature] * temporalBuffer[feature]
		}
		out[head] = sigmoid(logit, temperatures[head])
	}
	return indices
}
