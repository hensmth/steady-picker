package steady

import "math"

const (
	fnvOffset64 = uint64(14695981039346656037)
	fnvPrime64  = uint64(1099511628211)
)

// encodeInto builds an averaged hashed byte n-gram embedding without allocating.
func encodeInto(text string, table []float32, bucket, dim, minN, maxN int, out []float32) int {
	clear(out)
	if text == "" || bucket <= 0 || dim <= 0 || len(out) < dim {
		return 0
	}
	count := 0
	for start := 0; start < len(text); start++ {
		hash := fnvOffset64
		for n := 1; n <= maxN && start+n <= len(text); n++ {
			b := text[start+n-1]
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			hash ^= uint64(b)
			hash *= fnvPrime64
			if n < minN {
				continue
			}
			row := int(hash%uint64(bucket)) * dim
			for d := 0; d < dim; d++ {
				out[d] += table[row+d]
			}
			count++
		}
	}
	if count == 0 {
		return 0
	}
	scale := float32(1 / float64(count))
	for d := 0; d < dim; d++ {
		out[d] *= scale
		if math.IsNaN(float64(out[d])) || math.IsInf(float64(out[d]), 0) {
			out[d] = 0
		}
	}
	return count
}

func ngramBuckets(text string, bucket, minN, maxN int, dst []int) []int {
	dst = dst[:0]
	for start := 0; start < len(text); start++ {
		hash := fnvOffset64
		for n := 1; n <= maxN && start+n <= len(text); n++ {
			b := text[start+n-1]
			if b >= 'A' && b <= 'Z' {
				b += 'a' - 'A'
			}
			hash ^= uint64(b)
			hash *= fnvPrime64
			if n >= minN {
				dst = append(dst, int(hash%uint64(bucket)))
			}
		}
	}
	return dst
}
