package steady

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
)

const (
	modelMagic      uint32 = 0x53544459 // STDY
	modelVersion    uint32 = 5
	hybridVersion   uint32 = 4
	legacyVersion   uint32 = 3
	modelHeaderSize        = 32
	maxArtifactSize        = 64 << 20
	maxMetadataSize        = 1 << 20
	maxLabels              = 32
)

// Metadata is the canonical, immutable provenance carried by an artifact.
type Metadata struct {
	ArtifactFormat       int       `json:"artifact_format"`
	ModelID              string    `json:"model_id"`
	Task                 string    `json:"task"`
	Labels               []string  `json:"labels"`
	PolicyCompatibility  string    `json:"policy_compatibility"`
	MinN                 int       `json:"min_ngram"`
	MaxN                 int       `json:"max_ngram"`
	Bucket               int       `json:"bucket"`
	Dimension            int       `json:"dimension"`
	Epochs               int       `json:"epochs"`
	LearningRate         float64   `json:"learning_rate"`
	L2                   float64   `json:"l2"`
	Alpha                float64   `json:"alpha"`
	Seed                 uint64    `json:"seed"`
	SourceManifestSHA256 string    `json:"source_manifest_sha256"`
	TrainingCodeCommit   string    `json:"training_code_commit"`
	ArtifactSHA256       string    `json:"artifact_sha256,omitempty"`
	ModelFamily          string    `json:"model_family,omitempty"`
	FeatureSchema        string    `json:"feature_schema,omitempty"`
	Heads                []string  `json:"heads,omitempty"`
	WordMinN             int       `json:"word_min_ngram,omitempty"`
	WordMaxN             int       `json:"word_max_ngram,omitempty"`
	TemporalFeatures     int       `json:"temporal_features,omitempty"`
	PositiveClassWeights []float64 `json:"positive_class_weights,omitempty"`
	Tokenizer            string    `json:"tokenizer,omitempty"`
	Vocabulary           []string  `json:"vocabulary,omitempty"`
	MaxTokens            int       `json:"max_tokens,omitempty"`
	Layers               int       `json:"layers,omitempty"`
	AttentionHeads       int       `json:"attention_heads,omitempty"`
	HiddenSize           int       `json:"hidden_size,omitempty"`
	IntermediateSize     int       `json:"intermediate_size,omitempty"`
	Quantization         string    `json:"quantization,omitempty"`
	AuxiliaryHeads       []string  `json:"auxiliary_heads,omitempty"`
	TeacherEncoder       string    `json:"teacher_encoder,omitempty"`
	TrainingProvider     string    `json:"training_provider,omitempty"`
	TrainingModel        string    `json:"training_model,omitempty"`
	TrainingEffort       string    `json:"training_effort,omitempty"`
}

// Model contains immutable model weights and a pool of independent workspaces.
// It is safe for concurrent use and has no lifecycle-dependent Close method.
type Model struct {
	table            []float32
	weights          []float32
	bias             []float32
	temperature      float32
	quantiles        []float32
	thresholds       []float32
	bucket           int
	dim              int
	minN             int
	maxN             int
	metadata         Metadata
	workspaces       sync.Pool
	temperatures     []float32
	wordMinN         int
	wordMaxN         int
	temporalFeatures int
	semantic         *semanticModel
}

// Metadata returns a caller-owned copy of the artifact metadata.
func (m *Model) Metadata() Metadata {
	if m == nil {
		return Metadata{}
	}
	out := m.metadata
	out.Labels = slices.Clone(m.metadata.Labels)
	out.Heads = slices.Clone(m.metadata.Heads)
	out.PositiveClassWeights = slices.Clone(m.metadata.PositiveClassWeights)
	out.Vocabulary = slices.Clone(m.metadata.Vocabulary)
	out.AuxiliaryHeads = slices.Clone(m.metadata.AuxiliaryHeads)
	return out
}

// Load reads a strict v3, v4, or v5 artifact from path.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("steady: open model: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("steady: stat model: %w", err)
	}
	if info.Size() > maxArtifactSize {
		return nil, fmt.Errorf("steady: artifact exceeds %d-byte limit", maxArtifactSize)
	}
	if info.Size() < modelHeaderSize {
		return nil, errors.New("steady: model is smaller than its header")
	}
	data, err := io.ReadAll(io.LimitReader(f, maxArtifactSize+1))
	if err != nil {
		return nil, fmt.Errorf("steady: read model: %w", err)
	}
	return LoadBytes(data)
}

// LoadBytes parses a strict v3, v4, or v5 artifact and copies all model data.
func LoadBytes(input []byte) (*Model, error) {
	if len(input) > maxArtifactSize {
		return nil, fmt.Errorf("steady: artifact exceeds %d-byte limit", maxArtifactSize)
	}
	if len(input) < modelHeaderSize {
		return nil, errors.New("steady: model is smaller than its header")
	}
	if binary.LittleEndian.Uint32(input[:4]) != modelMagic {
		if binary.LittleEndian.Uint32(input[4:8]) == 2 {
			return nil, errors.New("steady: artifact format v2 is unsafe because label order is ambiguous; use steady-picker v0.1 or retrain as v4")
		}
		return nil, errors.New("steady: not a steady model file")
	}
	version := binary.LittleEndian.Uint32(input[4:8])
	if version == 2 {
		return nil, errors.New("steady: artifact format v2 is unsupported; use steady-picker v0.1 or retrain as v4")
	}
	if version != legacyVersion && version != hybridVersion && version != modelVersion {
		return nil, fmt.Errorf("steady: unsupported artifact format v%d", version)
	}
	if binary.LittleEndian.Uint32(input[12:16]) != 0 ||
		binary.LittleEndian.Uint64(input[24:32]) != 0 {
		return nil, errors.New("steady: non-canonical reserved header bytes")
	}
	metadataLen := int(binary.LittleEndian.Uint32(input[8:12]))
	payloadLen := int(binary.LittleEndian.Uint64(input[16:24]))
	if metadataLen <= 0 || metadataLen > maxMetadataSize {
		return nil, errors.New("steady: invalid metadata length")
	}
	if payloadLen <= 0 || modelHeaderSize+metadataLen+payloadLen != len(input) {
		return nil, errors.New("steady: invalid artifact payload length")
	}

	var metadata Metadata
	decoder := json.NewDecoder(bytes.NewReader(input[modelHeaderSize : modelHeaderSize+metadataLen]))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metadata); err != nil {
		return nil, fmt.Errorf("steady: decode metadata: %w", err)
	}
	if decoder.Decode(&struct{}{}) != io.EOF {
		return nil, errors.New("steady: metadata contains trailing JSON")
	}
	if metadata.ArtifactFormat != int(version) {
		return nil, errors.New("steady: header and metadata artifact formats differ")
	}
	if err := validateMetadata(metadata); err != nil {
		return nil, err
	}

	nLabels := len(metadata.Labels)
	if version == modelVersion {
		m, err := loadSemanticV5(
			input[modelHeaderSize+metadataLen:], metadata,
		)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(input)
		m.metadata.ArtifactSHA256 = hex.EncodeToString(sum[:])
		m.workspaces.New = func() any { return m.newWorkspace() }
		return m, nil
	}
	if version == hybridVersion {
		floatCount, ok := checkedV4PayloadFloats(
			metadata.Bucket, metadata.Dimension, metadata.TemporalFeatures, len(metadata.Heads),
		)
		if !ok || payloadLen != floatCount*4 {
			return nil, errors.New("steady: v4 payload dimensions do not match metadata")
		}
		values, err := decodeFloats(input[modelHeaderSize+metadataLen:], floatCount)
		if err != nil {
			return nil, err
		}
		pos := 0
		take := func(n int) []float32 {
			part := slices.Clone(values[pos : pos+n])
			pos += n
			return part
		}
		heads := len(metadata.Heads)
		table := take(metadata.Bucket * metadata.Dimension)
		weights := take(heads * (metadata.Bucket + metadata.Dimension + metadata.TemporalFeatures))
		bias := take(heads)
		temperatures := take(heads)
		quantiles := take(heads * 2)
		thresholds := take(heads)
		for _, value := range temperatures {
			if value <= 0 {
				return nil, errors.New("steady: temperatures must be greater than zero")
			}
		}
		for _, value := range append(slices.Clone(quantiles), thresholds...) {
			if value < 0 || value > 1 {
				return nil, errors.New("steady: quantiles and thresholds must be in [0,1]")
			}
		}
		sum := sha256.Sum256(input)
		metadata.ArtifactSHA256 = hex.EncodeToString(sum[:])
		m := &Model{
			table: table, weights: weights, bias: bias, temperatures: temperatures,
			quantiles: quantiles, thresholds: thresholds, bucket: metadata.Bucket,
			dim: metadata.Dimension, minN: metadata.MinN, maxN: metadata.MaxN,
			wordMinN: metadata.WordMinN, wordMaxN: metadata.WordMaxN,
			temporalFeatures: metadata.TemporalFeatures, metadata: metadata,
		}
		m.workspaces.New = func() any { return m.newWorkspace() }
		return m, nil
	}
	floatCount, ok := checkedPayloadFloats(metadata.Bucket, metadata.Dimension, nLabels)
	if !ok || payloadLen != floatCount*4 {
		return nil, errors.New("steady: payload dimensions do not match metadata")
	}
	payload := input[modelHeaderSize+metadataLen:]
	values, err := decodeFloats(payload, floatCount)
	if err != nil {
		return nil, err
	}
	pos := 0
	take := func(n int) []float32 {
		part := slices.Clone(values[pos : pos+n])
		pos += n
		return part
	}
	table := take(metadata.Bucket * metadata.Dimension)
	weights := take(nLabels * metadata.Dimension)
	bias := take(nLabels)
	temperature := take(1)[0]
	quantiles := take(nLabels)
	thresholds := take(nLabels)
	if temperature <= 0 {
		return nil, errors.New("steady: temperature must be finite and greater than zero")
	}
	for i := range nLabels {
		if quantiles[i] < 0 || quantiles[i] > 1 || thresholds[i] < 0 || thresholds[i] > 1 {
			return nil, errors.New("steady: quantiles and thresholds must be in [0,1]")
		}
	}
	sum := sha256.Sum256(input)
	metadata.ArtifactSHA256 = hex.EncodeToString(sum[:])
	m := &Model{
		table: table, weights: weights, bias: bias, temperature: temperature,
		quantiles: quantiles, thresholds: thresholds, bucket: metadata.Bucket,
		dim: metadata.Dimension, minN: metadata.MinN, maxN: metadata.MaxN,
		metadata: metadata,
	}
	m.workspaces.New = func() any { return m.newWorkspace() }
	return m, nil
}

func validateMetadata(m Metadata) error {
	if m.ArtifactSHA256 != "" {
		return errors.New("steady: artifact SHA-256 is loader-derived and must not appear in metadata")
	}
	if (m.ArtifactFormat != int(legacyVersion) &&
		m.ArtifactFormat != int(hybridVersion) &&
		m.ArtifactFormat != int(modelVersion)) ||
		m.ModelID != "settings-v2" ||
		m.Task != "video-duration-selection" {
		return errors.New("steady: invalid artifact identity metadata")
	}
	if !slices.Equal(m.Labels, []string{"short", "medium", "long"}) {
		return errors.New("steady: semantic label order must be exactly short, medium, long")
	}
	switch m.ArtifactFormat {
	case int(modelVersion):
		if m.ModelFamily != "distilled-semantic-transformer-v1" ||
			m.FeatureSchema != "wordpiece-transformer-temporal-aux-v1" ||
			!slices.Equal(m.Heads, []string{"short", "long"}) ||
			m.Tokenizer != "wordpiece-v1" ||
			len(m.Vocabulary) < 5 || len(m.Vocabulary) > 8192 ||
			m.MaxTokens != 96 || m.Layers != 2 ||
			m.AttentionHeads != 4 || m.HiddenSize != 128 ||
			m.IntermediateSize != 512 ||
			m.Quantization != "int8-per-output-channel-f32-layernorm" ||
			!slices.Equal(m.AuxiliaryHeads, []string{
				"beats_1", "beats_2", "beats_3",
				"ordered", "transformation", "dependent_actions",
			}) ||
			m.TeacherEncoder == "" || m.TrainingProvider == "" ||
			m.TrainingModel == "" || m.TrainingEffort == "" {
			return errors.New("steady: invalid v5 semantic model metadata")
		}
		if m.Bucket != 1 || m.Dimension != m.HiddenSize ||
			m.MinN != 1 || m.MaxN != 1 ||
			m.WordMinN != 0 || m.WordMaxN != 0 ||
			m.TemporalFeatures != 0 || len(m.PositiveClassWeights) != 2 {
			return errors.New("steady: invalid v5 compatibility dimensions")
		}
		if err := validateVocabulary(m.Vocabulary); err != nil {
			return err
		}
		for _, weight := range m.PositiveClassWeights {
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				return errors.New("steady: invalid v5 positive class weight")
			}
		}
	case int(hybridVersion):
		if m.ModelFamily != "dual-head-hybrid-v1" ||
			m.FeatureSchema != "hashed-char-word-linear+embedding+temporal-v1" ||
			!slices.Equal(m.Heads, []string{"short", "long"}) ||
			m.WordMinN != 1 || m.WordMaxN != 3 ||
			m.TemporalFeatures != temporalFeatureCount ||
			len(m.PositiveClassWeights) != 2 {
			return errors.New("steady: invalid v4 model-family metadata")
		}
		if m.Tokenizer != "" || len(m.Vocabulary) != 0 ||
			m.MaxTokens != 0 || m.Layers != 0 || m.AttentionHeads != 0 ||
			m.HiddenSize != 0 || m.IntermediateSize != 0 ||
			m.Quantization != "" || len(m.AuxiliaryHeads) != 0 ||
			m.TeacherEncoder != "" || m.TrainingProvider != "" ||
			m.TrainingModel != "" || m.TrainingEffort != "" {
			return errors.New("steady: v4 metadata contains v5 fields")
		}
		for _, weight := range m.PositiveClassWeights {
			if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
				return errors.New("steady: invalid v4 positive class weight")
			}
		}
	case int(legacyVersion):
		if m.ModelFamily != "" || m.FeatureSchema != "" || len(m.Heads) != 0 ||
			m.WordMinN != 0 || m.WordMaxN != 0 || m.TemporalFeatures != 0 ||
			len(m.PositiveClassWeights) != 0 || m.Tokenizer != "" ||
			len(m.Vocabulary) != 0 || m.MaxTokens != 0 || m.Layers != 0 ||
			m.AttentionHeads != 0 || m.HiddenSize != 0 ||
			m.IntermediateSize != 0 || m.Quantization != "" ||
			len(m.AuxiliaryHeads) != 0 || m.TeacherEncoder != "" ||
			m.TrainingProvider != "" || m.TrainingModel != "" ||
			m.TrainingEffort != "" {
			return errors.New("steady: v3 metadata contains newer-format fields")
		}
	}
	if m.PolicyCompatibility == "" || m.SourceManifestSHA256 == "" || m.TrainingCodeCommit == "" {
		return errors.New("steady: incomplete provenance metadata")
	}
	if len(m.PolicyCompatibility) > 128 || len(m.TrainingCodeCommit) > 256 {
		return errors.New("steady: provenance metadata is too long")
	}
	if m.Bucket < 1 || m.Bucket > 1_000_000 || m.Dimension < 1 || m.Dimension > 4096 ||
		m.MinN < 1 || m.MaxN < m.MinN || m.MaxN > 16 || m.Epochs < 1 ||
		m.Epochs > 10000 || m.LearningRate <= 0 || m.LearningRate > 10 ||
		m.L2 < 0 || m.L2 > 10 || m.Alpha <= 0 || m.Alpha >= 1 {
		return errors.New("steady: invalid model dimensions or hyperparameters")
	}
	if _, err := hex.DecodeString(m.SourceManifestSHA256); err != nil ||
		len(m.SourceManifestSHA256) != 64 ||
		strings.ToLower(m.SourceManifestSHA256) != m.SourceManifestSHA256 {
		return errors.New("steady: source manifest digest must be lowercase SHA-256 hex")
	}
	return nil
}

func checkedV4PayloadFloats(bucket, dim, temporal, heads int) (int, bool) {
	if bucket <= 0 || bucket > 1_000_000 || dim <= 0 || dim > 4096 ||
		temporal != temporalFeatureCount || heads != 2 {
		return 0, false
	}
	n64 := int64(bucket)*int64(dim) +
		int64(heads)*int64(bucket+dim+temporal) + int64(heads)*5
	if n64 <= 0 || n64 > int64(maxArtifactSize/4) {
		return 0, false
	}
	return int(n64), true
}

func decodeFloats(payload []byte, count int) ([]float32, error) {
	values := make([]float32, count)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
		if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
			return nil, fmt.Errorf("steady: non-finite numeric value at payload index %d", i)
		}
	}
	return values, nil
}

func checkedPayloadFloats(bucket, dim, labels int) (int, bool) {
	if bucket <= 0 || dim <= 0 || labels <= 0 || labels > maxLabels {
		return 0, false
	}
	n64 := int64(bucket)*int64(dim) + int64(labels)*int64(dim) + 1 + int64(labels)*3
	if n64 <= 0 || n64 > int64(maxArtifactSize/4) {
		return 0, false
	}
	return int(n64), true
}

func marshalModel(m *Model) ([]byte, error) {
	if m != nil && m.metadata.ArtifactFormat == int(modelVersion) {
		return marshalSemanticModel(m)
	}
	metadata := m.metadata
	metadata.ArtifactSHA256 = ""
	meta, err := json.Marshal(metadata)
	if err != nil {
		return nil, err
	}
	if len(meta) == 0 || len(meta) > maxMetadataSize {
		return nil, errors.New("steady: model metadata is too large")
	}
	var count int
	var ok bool
	var groups [][]float32
	if metadata.ArtifactFormat == int(hybridVersion) {
		count, ok = checkedV4PayloadFloats(
			m.bucket, m.dim, m.temporalFeatures, len(metadata.Heads),
		)
		groups = [][]float32{
			m.table, m.weights, m.bias, m.temperatures, m.quantiles, m.thresholds,
		}
	} else {
		count, ok = checkedPayloadFloats(m.bucket, m.dim, len(metadata.Labels))
		groups = [][]float32{m.table, m.weights, m.bias, {m.temperature}, m.quantiles, m.thresholds}
	}
	if !ok {
		return nil, errors.New("steady: model is too large")
	}
	payloadBytes := int64(count) * 4
	totalBytes := int64(modelHeaderSize) + int64(len(meta)) + payloadBytes
	if payloadBytes <= 0 || totalBytes > int64(maxArtifactSize) {
		return nil, errors.New("steady: model artifact is too large")
	}
	out := make([]byte, int(totalBytes))
	binary.LittleEndian.PutUint32(out[:4], modelMagic)
	binary.LittleEndian.PutUint32(out[4:8], uint32(metadata.ArtifactFormat))
	binary.LittleEndian.PutUint32(out[8:12], uint32(len(meta)))
	binary.LittleEndian.PutUint64(out[16:24], uint64(payloadBytes))
	copy(out[modelHeaderSize:], meta)
	pos := modelHeaderSize + len(meta)
	for _, group := range groups {
		for _, value := range group {
			binary.LittleEndian.PutUint32(out[pos:], math.Float32bits(value))
			pos += 4
		}
	}
	return out, nil
}
