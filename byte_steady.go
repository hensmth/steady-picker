// Package steady provides a zero-allocation text classification engine using
// byte-level n-gram hashing, logistic regression, Platt scaling, and conformal
// prediction. Designed for sub-millisecond CPU inference in pure Go.
package steady

import (
	"github.com/xDarkicex/memory"
)

// windows is the set of byte n-gram sizes used for multi-scale encoding.
var windows = [4]int{4, 8, 12, 16}

// Encode computes the averaged byteSteady embedding for raw text. The table
// parameter is a bucket × dim float32 matrix in row-major order. The returned
// slice is allocated from pool and has length dim.
func Encode(text []byte, table []float32, bucket, dim int, pool *memory.Pool) []float32 {
	hidden := memory.MustPoolSlice[float32](pool, dim)
	hidden = hidden[:dim]
	for i := range hidden {
		hidden[i] = 0
	}
	if len(text) == 0 {
		return hidden
	}
	var total int
	for _, n := range windows {
		if n > len(text) {
			continue
		}
		total += encodeScale(text, table, bucket, dim, n, hidden)
	}
	if total > 0 {
		invCount := 1.0 / float32(total)
		for i := range hidden {
			hidden[i] *= invCount
		}
	}
	return hidden
}

// encodeScale slides a window of size n across text, hashes each window modulo bucket,
// and accumulates the corresponding table row into hidden.
func encodeScale(text []byte, table []float32, bucket, dim, n int, hidden []float32) int {
	if n > len(text) {
		return 0
	}
	count := 0
	for i := 0; i <= len(text)-n; i++ {
		h := hashWindow(text[i : i+n])
		accumulateRow(table, bucket, dim, h, hidden)
		count++
	}
	return count
}

// hashWindow computes a 32-bit FNV-1a hash of data.
func hashWindow(data []byte) uint32 {
	var h uint32 = 2166136261
	for _, b := range data {
		h ^= uint32(b)
		h *= 16777619
	}
	return h
}

// accumulateRow adds the table row at index (h % bucket) to hidden.
func accumulateRow(table []float32, bucket, dim int, h uint32, hidden []float32) {
	row := int(h%uint32(bucket)) * dim
	for j := 0; j < dim; j++ {
		hidden[j] += table[row+j]
	}
}
