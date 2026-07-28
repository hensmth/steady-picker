package steady

import (
	"math"
	"unicode/utf8"
)

func (m *Model) semanticPredict(text string, work *workspace) bool {
	if m.semantic == nil || !utf8.ValidString(text) ||
		len(text) == 0 || len(text) > cap(work.textBytes) {
		return false
	}
	semantic := m.semantic
	count := semantic.tokenize(text, work)
	if count < 2 {
		return false
	}
	width := semantic.hidden
	x := work.semanticX[:count*width]
	for position, tokenID := range work.tokenIDs[:count] {
		tokenOffset := tokenID * width
		positionOffset := position * width
		tokenScale := semantic.tokenEmbeddings.scales[tokenID]
		positionScale := semantic.positionEmbeddings.scales[position]
		for index := range width {
			x[positionOffset+index] =
				float32(semantic.tokenEmbeddings.weights[tokenOffset+index])*tokenScale +
					float32(semantic.positionEmbeddings.weights[positionOffset+index])*positionScale
		}
		layerNormInPlace(
			x[positionOffset:positionOffset+width],
			semantic.embeddingNorm,
		)
	}

	q := work.semanticQ[:count*width]
	k := work.semanticK[:count*width]
	v := work.semanticV[:count*width]
	context := work.semanticY[:count*width]
	for _, layer := range semantic.layers {
		for position := range count {
			offset := position * width
			quantizedMatVec(q[offset:offset+width], layer.query, x[offset:offset+width])
			quantizedMatVec(k[offset:offset+width], layer.key, x[offset:offset+width])
			quantizedMatVec(v[offset:offset+width], layer.value, x[offset:offset+width])
		}
		clear(context)
		headWidth := width / semantic.attentionHeads
		scale := float32(1 / math.Sqrt(float64(headWidth)))
		for queryPosition := range count {
			for head := range semantic.attentionHeads {
				headStart := head * headWidth
				queryOffset := queryPosition*width + headStart
				scores := work.attention[:count]
				maxScore := float32(-math.MaxFloat32)
				for keyPosition := range count {
					keyOffset := keyPosition*width + headStart
					score := float32(0)
					for index := range headWidth {
						score += q[queryOffset+index] * k[keyOffset+index]
					}
					score *= scale
					scores[keyPosition] = score
					if score > maxScore {
						maxScore = score
					}
				}
				sum := float32(0)
				for index, score := range scores {
					value := float32(math.Exp(float64(score - maxScore)))
					scores[index] = value
					sum += value
				}
				outputOffset := queryPosition*width + headStart
				for keyPosition, score := range scores {
					weight := score / sum
					valueOffset := keyPosition*width + headStart
					for index := range headWidth {
						context[outputOffset+index] += weight * v[valueOffset+index]
					}
				}
			}
		}
		for position := range count {
			offset := position * width
			quantizedMatVec(
				work.semanticHidden[:width],
				layer.attentionOutput,
				context[offset:offset+width],
			)
			for index := range width {
				x[offset+index] += work.semanticHidden[index]
			}
			layerNormInPlace(x[offset:offset+width], layer.attentionNorm)

			quantizedMatVec(
				work.semanticFFN[:semantic.intermediate],
				layer.feedForwardIn,
				x[offset:offset+width],
			)
			for index, value := range work.semanticFFN[:semantic.intermediate] {
				work.semanticFFN[index] = gelu(value)
			}
			quantizedMatVec(
				work.semanticHidden[:width],
				layer.feedForwardOut,
				work.semanticFFN[:semantic.intermediate],
			)
			for index := range width {
				x[offset+index] += work.semanticHidden[index]
			}
			layerNormInPlace(x[offset:offset+width], layer.outputNorm)
		}
	}

	quantizedMatVec(
		work.logits[:len(m.metadata.Heads)],
		semantic.durationHeads,
		x[:width],
	)
	for index, logit := range work.logits[:len(m.metadata.Heads)] {
		work.probs[index] = semanticSigmoid(logit / m.temperatures[index])
	}
	return true
}

func quantizedMatVec(output []float32, matrix quantizedMatrix, input []float32) {
	for row := range matrix.rows {
		if matrix.scales[row] == 0 {
			if len(matrix.bias) != 0 {
				output[row] = matrix.bias[row]
			} else {
				output[row] = 0
			}
			continue
		}
		offset := row * matrix.cols
		sum := float32(0)
		for column := range matrix.cols {
			sum += float32(matrix.weights[offset+column]) * input[column]
		}
		value := sum * matrix.scales[row]
		if len(matrix.bias) != 0 {
			value += matrix.bias[row]
		}
		output[row] = value
	}
}

func layerNormInPlace(values []float32, weights layerNormWeights) {
	mean := float32(0)
	for _, value := range values {
		mean += value
	}
	mean /= float32(len(values))
	variance := float32(0)
	for _, value := range values {
		delta := value - mean
		variance += delta * delta
	}
	variance /= float32(len(values))
	inverse := float32(1 / math.Sqrt(float64(variance+1e-5)))
	for index, value := range values {
		values[index] = (value-mean)*inverse*weights.gamma[index] + weights.beta[index]
	}
}

func gelu(value float32) float32 {
	x := float64(value)
	return float32(0.5 * x * (1 + math.Tanh(
		math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x),
	)))
}

func semanticSigmoid(value float32) float32 {
	if value >= 0 {
		exp := float32(math.Exp(float64(-value)))
		return 1 / (1 + exp)
	}
	exp := float32(math.Exp(float64(value)))
	return exp / (1 + exp)
}

func (m *semanticModel) tokenize(text string, work *workspace) int {
	work.textBytes = work.textBytes[:len(text)]
	for index := range len(text) {
		value := text[index]
		if value >= 'A' && value <= 'Z' {
			value += 'a' - 'A'
		}
		work.textBytes[index] = value
	}
	tokens := work.tokenIDs
	count := 1
	tokens[0] = 2 // [CLS]
	for start := 0; start < len(work.textBytes) && count < m.maxTokens-1; {
		for start < len(work.textBytes) && isWordPieceSpace(work.textBytes[start]) {
			start++
		}
		if start == len(work.textBytes) {
			break
		}
		end := start + 1
		if isWordPieceASCII(work.textBytes[start]) {
			for end < len(work.textBytes) && isWordPieceASCII(work.textBytes[end]) {
				end++
			}
		} else if work.textBytes[start] >= utf8.RuneSelf {
			for end < len(work.textBytes) &&
				work.textBytes[end] >= utf8.RuneSelf {
				end++
			}
		}
		count = m.tokenizeWord(work.textBytes[start:end], tokens, count)
		start = end
	}
	tokens[count] = 3 // [SEP]
	return count + 1
}

func (m *semanticModel) tokenizeWord(word []byte, tokens []int, count int) int {
	if len(word) == 0 || len(word) > 100 || containsNonASCII(word) {
		if count < m.maxTokens-1 {
			tokens[count] = 1
			return count + 1
		}
		return count
	}
	original := count
	start := 0
	for start < len(word) && count < m.maxTokens-1 {
		foundID := -1
		foundEnd := start
		for end := len(word); end > start; end-- {
			if id, ok := m.lookupPiece(word[start:end], start != 0); ok {
				foundID, foundEnd = id, end
				break
			}
		}
		if foundID < 0 {
			tokens[original] = 1
			return original + 1
		}
		tokens[count] = foundID
		count++
		start = foundEnd
	}
	return count
}

func (m *semanticModel) lookupPiece(piece []byte, continuation bool) (int, bool) {
	hash := uint64(14695981039346656037)
	if continuation {
		hash ^= '#'
		hash *= 1099511628211
		hash ^= '#'
		hash *= 1099511628211
	}
	for _, value := range piece {
		hash ^= uint64(value)
		hash *= 1099511628211
	}
	for _, id := range m.vocabByHash[hash] {
		token := m.vocabulary[id]
		offset := 0
		if continuation {
			if len(token) < 2 || token[0] != '#' || token[1] != '#' {
				continue
			}
			offset = 2
		}
		if len(token)-offset != len(piece) {
			continue
		}
		matches := true
		for index, value := range piece {
			if token[offset+index] != value {
				matches = false
				break
			}
		}
		if matches {
			return id, true
		}
	}
	return 0, false
}

func isWordPieceSpace(value byte) bool {
	return value == ' ' || value == '\t' || value == '\r' || value == '\n'
}

func isWordPieceASCII(value byte) bool {
	return value >= 'a' && value <= 'z' ||
		value >= '0' && value <= '9' ||
		value == '_' || value == '\''
}

func containsNonASCII(values []byte) bool {
	for _, value := range values {
		if value >= utf8.RuneSelf {
			return true
		}
	}
	return false
}
