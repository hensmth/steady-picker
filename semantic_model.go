package steady

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf8"
)

const (
	semanticVocabSize = 8192
	semanticMaxTokens = 96
	semanticHidden    = 128
	semanticHeads     = 4
	semanticFFN       = 512
	semanticLayers    = 2
	semanticAuxHeads  = 6
)

type quantizedMatrix struct {
	rows    int
	cols    int
	weights []int8
	scales  []float32
	bias    []float32
}

type layerNormWeights struct {
	gamma []float32
	beta  []float32
}

type semanticLayer struct {
	query, key, value quantizedMatrix
	attentionOutput   quantizedMatrix
	attentionNorm     layerNormWeights
	feedForwardIn     quantizedMatrix
	feedForwardOut    quantizedMatrix
	outputNorm        layerNormWeights
}

type semanticModel struct {
	tokenEmbeddings    quantizedMatrix
	positionEmbeddings quantizedMatrix
	embeddingNorm      layerNormWeights
	layers             []semanticLayer
	durationHeads      quantizedMatrix
	auxiliaryHeads     quantizedMatrix
	vocabByHash        map[uint64][]int
	vocabulary         []string
	maxTokens          int
	hidden             int
	attentionHeads     int
	intermediate       int
}

func validateVocabulary(vocabulary []string) error {
	if len(vocabulary) != semanticVocabSize {
		return fmt.Errorf(
			"steady: v5 vocabulary must contain exactly %d tokens",
			semanticVocabSize,
		)
	}
	special := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
	if !slices.Equal(vocabulary[:len(special)], special) {
		return errors.New("steady: v5 vocabulary has invalid special-token order")
	}
	seen := make(map[string]struct{}, len(vocabulary))
	total := 0
	for _, token := range vocabulary {
		if token == "" || len(token) > 128 || !utf8.ValidString(token) ||
			strings.ContainsAny(token, "\x00\r\n\t") {
			return errors.New("steady: v5 vocabulary contains an invalid token")
		}
		total += len(token)
		if total > 512<<10 {
			return errors.New("steady: v5 vocabulary exceeds the byte limit")
		}
		if _, exists := seen[token]; exists {
			return errors.New("steady: v5 vocabulary contains duplicate tokens")
		}
		seen[token] = struct{}{}
	}
	return nil
}

func loadSemanticV5(payload []byte, metadata Metadata) (*Model, error) {
	expected, ok := semanticPayloadSize(metadata)
	if !ok || expected != len(payload) {
		return nil, errors.New("steady: v5 payload dimensions do not match metadata")
	}
	reader := semanticPayloadReader{data: payload}
	semantic := &semanticModel{
		maxTokens:      metadata.MaxTokens,
		hidden:         metadata.HiddenSize,
		attentionHeads: metadata.AttentionHeads,
		intermediate:   metadata.IntermediateSize,
		vocabulary:     slices.Clone(metadata.Vocabulary),
	}
	var err error
	if semantic.tokenEmbeddings, err = reader.matrix(
		len(metadata.Vocabulary), metadata.HiddenSize, false,
	); err != nil {
		return nil, err
	}
	if semantic.positionEmbeddings, err = reader.matrix(
		metadata.MaxTokens, metadata.HiddenSize, false,
	); err != nil {
		return nil, err
	}
	if semantic.embeddingNorm, err = reader.layerNorm(metadata.HiddenSize); err != nil {
		return nil, err
	}
	semantic.layers = make([]semanticLayer, metadata.Layers)
	for index := range semantic.layers {
		layer := &semantic.layers[index]
		if layer.query, err = reader.matrix(
			metadata.HiddenSize, metadata.HiddenSize, true,
		); err != nil {
			return nil, err
		}
		if layer.key, err = reader.matrix(
			metadata.HiddenSize, metadata.HiddenSize, true,
		); err != nil {
			return nil, err
		}
		if layer.value, err = reader.matrix(
			metadata.HiddenSize, metadata.HiddenSize, true,
		); err != nil {
			return nil, err
		}
		if layer.attentionOutput, err = reader.matrix(
			metadata.HiddenSize, metadata.HiddenSize, true,
		); err != nil {
			return nil, err
		}
		if layer.attentionNorm, err = reader.layerNorm(metadata.HiddenSize); err != nil {
			return nil, err
		}
		if layer.feedForwardIn, err = reader.matrix(
			metadata.IntermediateSize, metadata.HiddenSize, true,
		); err != nil {
			return nil, err
		}
		if layer.feedForwardOut, err = reader.matrix(
			metadata.HiddenSize, metadata.IntermediateSize, true,
		); err != nil {
			return nil, err
		}
		if layer.outputNorm, err = reader.layerNorm(metadata.HiddenSize); err != nil {
			return nil, err
		}
	}
	if semantic.durationHeads, err = reader.matrix(
		len(metadata.Heads), metadata.HiddenSize, true,
	); err != nil {
		return nil, err
	}
	if semantic.auxiliaryHeads, err = reader.matrix(
		len(metadata.AuxiliaryHeads), metadata.HiddenSize, true,
	); err != nil {
		return nil, err
	}
	temperatures, err := reader.floats(len(metadata.Heads), true)
	if err != nil {
		return nil, err
	}
	quantiles, err := reader.floats(len(metadata.Heads)*2, false)
	if err != nil {
		return nil, err
	}
	thresholds, err := reader.floats(len(metadata.Heads), false)
	if err != nil {
		return nil, err
	}
	for _, value := range temperatures {
		if value <= 0 {
			return nil, errors.New("steady: v5 temperatures must be greater than zero")
		}
	}
	for _, value := range append(slices.Clone(quantiles), thresholds...) {
		if value < 0 || value > 1 {
			return nil, errors.New("steady: v5 quantiles and thresholds must be in [0,1]")
		}
	}
	if reader.position != len(payload) {
		return nil, errors.New("steady: v5 payload contains trailing bytes")
	}
	semantic.buildVocabularyIndex()
	return &Model{
		bucket:       1,
		dim:          metadata.HiddenSize,
		minN:         1,
		maxN:         1,
		metadata:     metadata,
		temperatures: temperatures,
		quantiles:    quantiles,
		thresholds:   thresholds,
		semantic:     semantic,
	}, nil
}

func semanticPayloadSize(metadata Metadata) (int, bool) {
	if metadata.ArtifactFormat != int(modelVersion) ||
		len(metadata.Vocabulary) != semanticVocabSize ||
		metadata.MaxTokens != semanticMaxTokens ||
		metadata.HiddenSize != semanticHidden ||
		metadata.AttentionHeads != semanticHeads ||
		metadata.IntermediateSize != semanticFFN ||
		metadata.Layers != semanticLayers ||
		len(metadata.Heads) != 2 ||
		len(metadata.AuxiliaryHeads) != semanticAuxHeads {
		return 0, false
	}
	matrix := func(rows, cols int, bias bool) int64 {
		size := int64(rows)*int64(cols) + int64(rows)*4
		if bias {
			size += int64(rows) * 4
		}
		return size
	}
	n := matrix(len(metadata.Vocabulary), metadata.HiddenSize, false)
	n += matrix(metadata.MaxTokens, metadata.HiddenSize, false)
	n += int64(metadata.HiddenSize * 8)
	for range metadata.Layers {
		n += matrix(metadata.HiddenSize, metadata.HiddenSize, true) * 4
		n += int64(metadata.HiddenSize * 8)
		n += matrix(metadata.IntermediateSize, metadata.HiddenSize, true)
		n += matrix(metadata.HiddenSize, metadata.IntermediateSize, true)
		n += int64(metadata.HiddenSize * 8)
	}
	n += matrix(len(metadata.Heads), metadata.HiddenSize, true)
	n += matrix(len(metadata.AuxiliaryHeads), metadata.HiddenSize, true)
	n += int64((len(metadata.Heads) + len(metadata.Heads)*2 + len(metadata.Heads)) * 4)
	if n <= 0 || n > int64(maxArtifactSize) || n > int64(^uint(0)>>1) {
		return 0, false
	}
	return int(n), true
}

type semanticPayloadReader struct {
	data     []byte
	position int
}

func (r *semanticPayloadReader) matrix(rows, cols int, bias bool) (quantizedMatrix, error) {
	weightCount := rows * cols
	if rows <= 0 || cols <= 0 || weightCount/rows != cols ||
		r.position > len(r.data)-weightCount {
		return quantizedMatrix{}, errors.New("steady: truncated v5 quantized matrix")
	}
	weights := make([]int8, weightCount)
	for index, value := range r.data[r.position : r.position+weightCount] {
		weights[index] = int8(value)
	}
	r.position += weightCount
	scales, err := r.floats(rows, false)
	if err != nil {
		return quantizedMatrix{}, err
	}
	for _, scale := range scales {
		if scale < 0 {
			return quantizedMatrix{}, errors.New("steady: v5 quantization scale is negative")
		}
	}
	var biases []float32
	if bias {
		biases, err = r.floats(rows, false)
		if err != nil {
			return quantizedMatrix{}, err
		}
	}
	return quantizedMatrix{
		rows: rows, cols: cols, weights: weights, scales: scales, bias: biases,
	}, nil
}

func (r *semanticPayloadReader) layerNorm(size int) (layerNormWeights, error) {
	gamma, err := r.floats(size, false)
	if err != nil {
		return layerNormWeights{}, err
	}
	beta, err := r.floats(size, false)
	if err != nil {
		return layerNormWeights{}, err
	}
	return layerNormWeights{gamma: gamma, beta: beta}, nil
}

func (r *semanticPayloadReader) floats(count int, positive bool) ([]float32, error) {
	if count < 0 || count > (len(r.data)-r.position)/4 {
		return nil, errors.New("steady: truncated v5 float payload")
	}
	values := make([]float32, count)
	for index := range values {
		value := math.Float32frombits(binary.LittleEndian.Uint32(
			r.data[r.position+index*4:],
		))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) ||
			(positive && value <= 0) {
			return nil, errors.New("steady: invalid numeric value in v5 payload")
		}
		values[index] = value
	}
	r.position += count * 4
	return values, nil
}

func (m *semanticModel) buildVocabularyIndex() {
	m.vocabByHash = make(map[uint64][]int, len(m.vocabulary))
	for id, token := range m.vocabulary {
		hash := wordPieceHashString(token)
		m.vocabByHash[hash] = append(m.vocabByHash[hash], id)
	}
}

func wordPieceHashString(text string) uint64 {
	hash := uint64(14695981039346656037)
	for index := range len(text) {
		hash ^= uint64(text[index])
		hash *= 1099511628211
	}
	return hash
}

func marshalSemanticModel(model *Model) ([]byte, error) {
	if model == nil || model.semantic == nil {
		return nil, errors.New("steady: semantic model is nil")
	}
	metadata := model.metadata
	metadata.ArtifactSHA256 = ""
	if err := validateMetadata(metadata); err != nil {
		return nil, err
	}
	payloadSize, ok := semanticPayloadSize(metadata)
	if !ok {
		return nil, errors.New("steady: invalid semantic payload dimensions")
	}
	meta, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("steady: encode v5 metadata: %w", err)
	}
	total := modelHeaderSize + len(meta) + payloadSize
	if len(meta) == 0 || len(meta) > maxMetadataSize || total > maxArtifactSize {
		return nil, errors.New("steady: v5 artifact exceeds its size limit")
	}
	output := make([]byte, total)
	binary.LittleEndian.PutUint32(output[:4], modelMagic)
	binary.LittleEndian.PutUint32(output[4:8], modelVersion)
	binary.LittleEndian.PutUint32(output[8:12], uint32(len(meta)))
	binary.LittleEndian.PutUint64(output[16:24], uint64(payloadSize))
	copy(output[modelHeaderSize:], meta)
	writer := semanticPayloadWriter{
		data:     output,
		position: modelHeaderSize + len(meta),
	}
	semantic := model.semantic
	if err := writer.matrix(semantic.tokenEmbeddings, false); err != nil {
		return nil, err
	}
	if err := writer.matrix(semantic.positionEmbeddings, false); err != nil {
		return nil, err
	}
	if err := writer.layerNorm(semantic.embeddingNorm, metadata.HiddenSize); err != nil {
		return nil, err
	}
	for _, layer := range semantic.layers {
		for _, matrix := range []quantizedMatrix{
			layer.query, layer.key, layer.value, layer.attentionOutput,
		} {
			if err := writer.matrix(matrix, true); err != nil {
				return nil, err
			}
		}
		if err := writer.layerNorm(layer.attentionNorm, metadata.HiddenSize); err != nil {
			return nil, err
		}
		if err := writer.matrix(layer.feedForwardIn, true); err != nil {
			return nil, err
		}
		if err := writer.matrix(layer.feedForwardOut, true); err != nil {
			return nil, err
		}
		if err := writer.layerNorm(layer.outputNorm, metadata.HiddenSize); err != nil {
			return nil, err
		}
	}
	if err := writer.matrix(semantic.durationHeads, true); err != nil {
		return nil, err
	}
	if err := writer.matrix(semantic.auxiliaryHeads, true); err != nil {
		return nil, err
	}
	if err := writer.floats(model.temperatures); err != nil {
		return nil, err
	}
	if err := writer.floats(model.quantiles); err != nil {
		return nil, err
	}
	if err := writer.floats(model.thresholds); err != nil {
		return nil, err
	}
	if writer.position != len(output) {
		return nil, errors.New("steady: v5 serialization length mismatch")
	}
	return output, nil
}

type semanticPayloadWriter struct {
	data     []byte
	position int
}

func (w *semanticPayloadWriter) matrix(matrix quantizedMatrix, bias bool) error {
	if matrix.rows <= 0 || matrix.cols <= 0 ||
		len(matrix.weights) != matrix.rows*matrix.cols ||
		len(matrix.scales) != matrix.rows ||
		(bias && len(matrix.bias) != matrix.rows) ||
		(!bias && len(matrix.bias) != 0) ||
		w.position > len(w.data)-len(matrix.weights) {
		return errors.New("steady: invalid v5 matrix for serialization")
	}
	for index, value := range matrix.weights {
		w.data[w.position+index] = byte(value)
	}
	w.position += len(matrix.weights)
	if err := w.floats(matrix.scales); err != nil {
		return err
	}
	if bias {
		return w.floats(matrix.bias)
	}
	return nil
}

func (w *semanticPayloadWriter) layerNorm(weights layerNormWeights, size int) error {
	if len(weights.gamma) != size || len(weights.beta) != size {
		return errors.New("steady: invalid v5 layer norm for serialization")
	}
	if err := w.floats(weights.gamma); err != nil {
		return err
	}
	return w.floats(weights.beta)
}

func (w *semanticPayloadWriter) floats(values []float32) error {
	if w.position > len(w.data)-len(values)*4 {
		return errors.New("steady: v5 serialization exceeds payload")
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("steady: cannot serialize non-finite v5 value")
		}
		binary.LittleEndian.PutUint32(w.data[w.position:], math.Float32bits(value))
		w.position += 4
	}
	return nil
}
