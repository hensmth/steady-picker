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
	modelVersion    uint32 = 3
	modelHeaderSize        = 32
	maxArtifactSize        = 64 << 20
	maxMetadataSize        = 1 << 20
	maxLabels              = 32
)

// Metadata is the canonical, immutable provenance carried by a v3 artifact.
type Metadata struct {
	ArtifactFormat       int      `json:"artifact_format"`
	ModelID              string   `json:"model_id"`
	Task                 string   `json:"task"`
	Labels               []string `json:"labels"`
	PolicyCompatibility  string   `json:"policy_compatibility"`
	MinN                 int      `json:"min_ngram"`
	MaxN                 int      `json:"max_ngram"`
	Bucket               int      `json:"bucket"`
	Dimension            int      `json:"dimension"`
	Epochs               int      `json:"epochs"`
	LearningRate         float64  `json:"learning_rate"`
	L2                   float64  `json:"l2"`
	Alpha                float64  `json:"alpha"`
	Seed                 uint64   `json:"seed"`
	SourceManifestSHA256 string   `json:"source_manifest_sha256"`
	TrainingCodeCommit   string   `json:"training_code_commit"`
	ArtifactSHA256       string   `json:"artifact_sha256,omitempty"`
}

// Model contains immutable model weights and a pool of independent workspaces.
// It is safe for concurrent use and has no lifecycle-dependent Close method.
type Model struct {
	table       []float32
	weights     []float32
	bias        []float32
	temperature float32
	quantiles   []float32
	thresholds  []float32
	bucket      int
	dim         int
	minN        int
	maxN        int
	metadata    Metadata
	workspaces  sync.Pool
}

// Metadata returns a caller-owned copy of the artifact metadata.
func (m *Model) Metadata() Metadata {
	if m == nil {
		return Metadata{}
	}
	out := m.metadata
	out.Labels = slices.Clone(m.metadata.Labels)
	return out
}

// Load reads a strict v3 artifact from path.
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

// LoadBytes parses a strict v3 artifact and copies all model data.
func LoadBytes(input []byte) (*Model, error) {
	if len(input) > maxArtifactSize {
		return nil, fmt.Errorf("steady: artifact exceeds %d-byte limit", maxArtifactSize)
	}
	if len(input) < modelHeaderSize {
		return nil, errors.New("steady: model is smaller than its header")
	}
	if binary.LittleEndian.Uint32(input[:4]) != modelMagic {
		if binary.LittleEndian.Uint32(input[4:8]) == 2 {
			return nil, errors.New("steady: artifact format v2 is unsafe because label order is ambiguous; use steady-picker v0.1 or retrain as v3")
		}
		return nil, errors.New("steady: not a steady model file")
	}
	version := binary.LittleEndian.Uint32(input[4:8])
	if version == 2 {
		return nil, errors.New("steady: artifact format v2 is unsupported; use steady-picker v0.1 or retrain as v3")
	}
	if version != modelVersion {
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
	if err := validateMetadata(metadata); err != nil {
		return nil, err
	}

	nLabels := len(metadata.Labels)
	floatCount, ok := checkedPayloadFloats(metadata.Bucket, metadata.Dimension, nLabels)
	if !ok || payloadLen != floatCount*4 {
		return nil, errors.New("steady: payload dimensions do not match metadata")
	}
	values := make([]float32, floatCount)
	payload := input[modelHeaderSize+metadataLen:]
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
		if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
			return nil, fmt.Errorf("steady: non-finite numeric value at payload index %d", i)
		}
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
	if m.ArtifactFormat != int(modelVersion) || m.ModelID != "settings-v2" ||
		m.Task != "video-duration-selection" {
		return errors.New("steady: invalid artifact identity metadata")
	}
	if !slices.Equal(m.Labels, []string{"short", "medium", "long"}) {
		return errors.New("steady: semantic label order must be exactly short, medium, long")
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
	metadata := m.metadata
	metadata.ArtifactSHA256 = ""
	meta, err := json.Marshal(metadata)
	if err != nil {
		return nil, err
	}
	if len(meta) == 0 || len(meta) > maxMetadataSize {
		return nil, errors.New("steady: model metadata is too large")
	}
	count, ok := checkedPayloadFloats(m.bucket, m.dim, len(metadata.Labels))
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
	binary.LittleEndian.PutUint32(out[4:8], modelVersion)
	binary.LittleEndian.PutUint32(out[8:12], uint32(len(meta)))
	binary.LittleEndian.PutUint64(out[16:24], uint64(payloadBytes))
	copy(out[modelHeaderSize:], meta)
	pos := modelHeaderSize + len(meta)
	for _, group := range [][]float32{m.table, m.weights, m.bias, {m.temperature}, m.quantiles, m.thresholds} {
		for _, value := range group {
			binary.LittleEndian.PutUint32(out[pos:], math.Float32bits(value))
			pos += 4
		}
	}
	return out, nil
}
