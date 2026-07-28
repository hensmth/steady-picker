package steady

import (
	"math"
)

const temporalFeatureCount = 32

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
	var older, previous uint64
	wordCount := 0
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
		if wordCount >= 1 {
			pair := previous
			pair ^= hash + 0x9e3779b97f4a7c15 + (pair << 6) + (pair >> 2)
			dst = append(dst, int(pair%uint64(bucket)))
		}
		if wordCount >= 2 {
			triple := older
			triple ^= previous + 0x9e3779b97f4a7c15 + (triple << 6) + (triple >> 2)
			triple ^= hash + 0x517cc1b727220a95 + (triple << 6) + (triple >> 2)
			dst = append(dst, int(triple%uint64(bucket)))
		}
		older, previous, wordCount, start = previous, hash, wordCount+1, end
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
		{"transform", "turns into", "turn into", "turning into", "morph", "becomes "},
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
		{"over time", "life cycle", "season", "years ", "ages "},
		{"begin", "start", "finish", " end ", "ends "},
		{"create", "draw", "paint", "carve", "sculpt"},
		{"reveal", "discover", "uncover", "emerge"},
		{"launch", "takeoff", "take off", "land ", "landing"},
		{"weather", "storm", "sunrise", "sunset", "day to night"},
		{"enter", "encounter", "interview", "escape", "chase"},
		{"stages", "steps", "sequence", "process", "progression"},
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
	words, gerunds, punctuation := 0, 0, 0
	for start := 0; start < len(text); {
		if text[start] == ',' || text[start] == ';' || text[start] == ':' {
			punctuation++
		}
		if !asciiWord(text[start]) {
			start++
			continue
		}
		end := start
		for end < len(text) && asciiWord(text[end]) {
			end++
		}
		words++
		if end-start > 4 && equalFoldSuffix(text[start:end], "ing") {
			gerunds++
		}
		start = end
	}
	active := 0
	for _, value := range out[:24] {
		if value > 0 {
			active++
		}
	}
	out[24] = min(float32(words)/64, 1)
	out[25] = min(float32(punctuation)/8, 1)
	out[26] = min(float32(countFold(text, " and "))/6, 1)
	out[27] = min(float32(gerunds)/6, 1)
	out[28] = min(float32(countFold(text, " into ")+countFold(text, " through "))/4, 1)
	out[29] = min(float32(len(text))/512, 1)
	out[30] = min(float32(active)/8, 1)
	out[31] = min(float32(
		countFold(text, " then ")+countFold(text, " next ")+
			countFold(text, " finally ")+countFold(text, " followed by "),
	)/4, 1)
}

func equalFoldSuffix(text, suffix string) bool {
	if len(text) < len(suffix) {
		return false
	}
	start := len(text) - len(suffix)
	for index := range len(suffix) {
		a, b := text[start+index], suffix[index]
		if a >= 'A' && a <= 'Z' {
			a += 'a' - 'A'
		}
		if a != b {
			return false
		}
	}
	return true
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
		"transform", "turns into", "turn into", "turning into", "morph", "becomes ",
		"evolv", "grows into", "time-lapse", "timelapse", "gradually", "eventually",
		"over time", "life cycle", "through the seasons", "through seasons", "stages",
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
	embeddingScale := float32(0)
	sparseScale := float32(0)
	if len(indices) > 0 {
		embeddingScale = float32(1 / float64(len(indices)))
		sparseScale = float32(1 / math.Sqrt(float64(len(indices))))
		for _, index := range indices {
			base := index * dimension
			for feature := 0; feature < dimension; feature++ {
				hidden[feature] += table[base+feature]
			}
		}
		for feature := range dimension {
			hidden[feature] *= embeddingScale
		}
	}
	stride := bucket + dimension + temporal
	for head := range bias {
		logit := bias[head]
		base := head * stride
		for _, index := range indices {
			logit += weights[base+index] * sparseScale
		}
		for feature := 0; feature < dimension; feature++ {
			logit += weights[base+bucket+feature] * hidden[feature]
		}
		for feature := 0; feature < temporal; feature++ {
			logit += weights[base+bucket+dimension+feature] * temporalBuffer[feature]
		}
		out[head] = sigmoid(logit, temperatures[head])
	}
	return indices
}
