package steady

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
)

// TrainConfig describes a deterministic, single-threaded v3 training run.
// All four datasets must have been frozen before training.
type TrainConfig struct {
	TrainInput                  string
	ProbabilityCalibrationInput string
	ConformalCalibrationInput   string
	ThresholdDevelopmentInput   string
	Output                      string
	Bucket                      int
	Dimension                   int
	MinN                        int
	MaxN                        int
	Epochs                      int
	LearningRate                float32
	L2                          float32
	Alpha                       float64
	Seed                        uint64
	SourceManifestSHA256        string
	TrainingCodeCommit          string
	PositiveWeightScale         float32
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Bucket: 10_000, Dimension: 64, MinN: 3, MaxN: 5, Epochs: 25,
		LearningRate: 0.05, L2: 0.0001, Alpha: 0.05, Seed: 42,
		PositiveWeightScale: 1,
	}
}

type example struct {
	text  string
	label int
}

// Train fits a multinomial softmax model, calibrates it on disjoint datasets,
// freezes selective thresholds, and writes deterministic v3 bytes.
func trainLegacy(cfg TrainConfig) error {
	if err := validateTrainConfig(cfg); err != nil {
		return err
	}
	labels := []string{"short", "medium", "long"}
	train, err := parseExamples(cfg.TrainInput, labels)
	if err != nil {
		return fmt.Errorf("steady: parse training data: %w", err)
	}
	probability, err := parseExamples(cfg.ProbabilityCalibrationInput, labels)
	if err != nil {
		return fmt.Errorf("steady: parse probability calibration data: %w", err)
	}
	conformal, err := parseExamples(cfg.ConformalCalibrationInput, labels)
	if err != nil {
		return fmt.Errorf("steady: parse conformal calibration data: %w", err)
	}
	development, err := parseExamples(cfg.ThresholdDevelopmentInput, labels)
	if err != nil {
		return fmt.Errorf("steady: parse threshold development data: %w", err)
	}
	if len(train) == 0 || len(probability) == 0 || len(conformal) == 0 || len(development) == 0 {
		return errors.New("steady: every frozen training/calibration/development split must be non-empty")
	}

	table := deterministicValues(cfg.Bucket*cfg.Dimension, cfg.Seed^0x91e10da5)
	weights := deterministicValues(len(labels)*cfg.Dimension, cfg.Seed^0xd1b54a32)
	bias := make([]float32, len(labels))
	hidden := make([]float32, cfg.Dimension)
	logits := make([]float32, len(labels))
	probs := make([]float32, len(labels))
	gradientHidden := make([]float32, cfg.Dimension)
	ngramRows := make([]int, 0, 4096)
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		for _, ex := range train {
			count := encodeInto(ex.text, table, cfg.Bucket, cfg.Dimension, cfg.MinN, cfg.MaxN, hidden)
			if count == 0 {
				continue
			}
			predictLogits(hidden, weights, bias, logits, cfg.Dimension)
			softmax(logits, probs, 1)
			clear(gradientHidden)
			for label := range labels {
				gradient := probs[label]
				if label == ex.label {
					gradient--
				}
				base := label * cfg.Dimension
				for d := 0; d < cfg.Dimension; d++ {
					gradientHidden[d] += gradient * weights[base+d]
					weights[base+d] -= cfg.LearningRate * (gradient*hidden[d] + cfg.L2*weights[base+d])
				}
				bias[label] -= cfg.LearningRate * gradient
			}
			ngramRows = ngramBuckets(ex.text, cfg.Bucket, cfg.MinN, cfg.MaxN, ngramRows)
			scale := cfg.LearningRate / float32(count)
			for _, row := range ngramRows {
				base := row * cfg.Dimension
				for d := 0; d < cfg.Dimension; d++ {
					table[base+d] -= scale * gradientHidden[d]
				}
			}
		}
	}

	model := &Model{
		table: table, weights: weights, bias: bias, temperature: 1,
		quantiles: make([]float32, len(labels)), thresholds: []float32{1, 1, 1},
		bucket: cfg.Bucket, dim: cfg.Dimension, minN: cfg.MinN, maxN: cfg.MaxN,
		metadata: Metadata{
			ArtifactFormat: 3, ModelID: "settings-v2", Task: "video-duration-selection",
			Labels: labels, PolicyCompatibility: ProfileQuotaSafeV2,
			MinN: cfg.MinN, MaxN: cfg.MaxN, Bucket: cfg.Bucket, Dimension: cfg.Dimension,
			Epochs: cfg.Epochs, LearningRate: float64(cfg.LearningRate), L2: float64(cfg.L2),
			Alpha: cfg.Alpha, Seed: cfg.Seed, SourceManifestSHA256: cfg.SourceManifestSHA256,
			TrainingCodeCommit: cfg.TrainingCodeCommit,
		},
	}
	model.temperature = fitTemperature(model, probability)
	model.quantiles = fitClassQuantiles(model, conformal, cfg.Alpha)
	model.thresholds = fitPrecisionThresholds(model, development)
	artifact, err := marshalModel(model)
	if err != nil {
		return err
	}
	if err := os.WriteFile(cfg.Output, artifact, 0o644); err != nil {
		return fmt.Errorf("steady: write artifact: %w", err)
	}
	return nil
}

func validateTrainConfig(cfg TrainConfig) error {
	if cfg.TrainInput == "" || cfg.ProbabilityCalibrationInput == "" ||
		cfg.ConformalCalibrationInput == "" || cfg.ThresholdDevelopmentInput == "" || cfg.Output == "" {
		return errors.New("steady: all split paths and output are required")
	}
	meta := Metadata{
		ArtifactFormat: 3, ModelID: "settings-v2", Task: "video-duration-selection",
		Labels: []string{"short", "medium", "long"}, PolicyCompatibility: ProfileQuotaSafeV2,
		MinN: cfg.MinN, MaxN: cfg.MaxN, Bucket: cfg.Bucket, Dimension: cfg.Dimension,
		Epochs: cfg.Epochs, LearningRate: float64(cfg.LearningRate), L2: float64(cfg.L2),
		Alpha: cfg.Alpha, SourceManifestSHA256: cfg.SourceManifestSHA256,
		TrainingCodeCommit: cfg.TrainingCodeCommit,
	}
	return validateMetadata(meta)
}

func parseExamples(path string, labels []string) ([]example, error) {
	index := make(map[string]int, len(labels))
	for i, label := range labels {
		index[label] = i
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), maxPromptBytes+128)
	var out []example
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 || !strings.HasPrefix(parts[0], "__label__") {
			return nil, fmt.Errorf("malformed labelled row %d", len(out)+1)
		}
		label, ok := index[strings.TrimPrefix(parts[0], "__label__")]
		if !ok || strings.TrimSpace(parts[1]) == "" {
			return nil, fmt.Errorf("invalid label or prompt at row %d", len(out)+1)
		}
		out = append(out, example{strings.TrimSpace(parts[1]), label})
	}
	return out, scanner.Err()
}

func deterministicValues(count int, seed uint64) []float32 {
	out := make([]float32, count)
	state := seed
	for i := range out {
		state ^= state >> 12
		state ^= state << 25
		state ^= state >> 27
		value := state * 2685821657736338717
		out[i] = (float32(value>>40)/float32(1<<24)*2 - 1) * 0.01
	}
	return out
}

func probabilitiesFor(model *Model, examples []example, temperature float32) [][]float32 {
	out := make([][]float32, len(examples))
	hidden := make([]float32, model.dim)
	logits := make([]float32, len(model.metadata.Labels))
	for i, ex := range examples {
		encodeInto(ex.text, model.table, model.bucket, model.dim, model.minN, model.maxN, hidden)
		predictLogits(hidden, model.weights, model.bias, logits, model.dim)
		out[i] = make([]float32, len(logits))
		softmax(logits, out[i], temperature)
	}
	return out
}

func fitTemperature(model *Model, examples []example) float32 {
	bestTemperature, bestLoss := float32(1), math.Inf(1)
	for step := 1; step <= 200; step++ {
		temperature := float32(step) / 20
		probabilities := probabilitiesFor(model, examples, temperature)
		var loss float64
		for i, ex := range examples {
			loss -= math.Log(math.Max(float64(probabilities[i][ex.label]), 1e-12))
		}
		if loss < bestLoss {
			bestLoss, bestTemperature = loss, temperature
		}
	}
	return bestTemperature
}

func fitClassQuantiles(model *Model, examples []example, alpha float64) []float32 {
	probabilities := probabilitiesFor(model, examples, model.temperature)
	scores := make([][]float32, len(model.metadata.Labels))
	for i, ex := range examples {
		scores[ex.label] = append(scores[ex.label], 1-probabilities[i][ex.label])
	}
	out := make([]float32, len(scores))
	for label := range scores {
		out[label] = conformalQuantile(scores[label], alpha)
	}
	return out
}

func fitPrecisionThresholds(model *Model, examples []example) []float32 {
	probabilities := probabilitiesFor(model, examples, model.temperature)
	thresholds := []float32{1, 1, 1}
	required := map[int]float64{0: 0.95, 2: 0.98}
	for label, minimumPrecision := range required {
		for step := 160; step <= 199; step++ {
			threshold := float32(step) * 0.005
			accepted, correct := 0, 0
			for i, ex := range examples {
				best := slices.Index(probabilities[i], slices.Max(probabilities[i]))
				kinds := predictionKinds(
					probabilities[i],
					model.metadata.Labels,
					model.quantiles,
					nil,
				)
				if len(kinds) == 1 && best == label && kinds[0] == label &&
					probabilities[i][label] >= threshold {
					accepted++
					if ex.label == label {
						correct++
					}
				}
			}
			if accepted > 0 && float64(correct)/float64(accepted) >= minimumPrecision {
				thresholds[label] = threshold
				break
			}
		}
	}
	return thresholds
}
