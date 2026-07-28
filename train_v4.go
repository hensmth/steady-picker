package steady

import (
	"errors"
	"fmt"
	"math"
	"os"
)

const maxReleaseArtifactSize = 32 << 20

// Train fits the v4 class-balanced dual-head sparse model.
func Train(cfg TrainConfig) error {
	if cfg.TrainInput == "" || cfg.ProbabilityCalibrationInput == "" ||
		cfg.ConformalCalibrationInput == "" || cfg.ThresholdDevelopmentInput == "" ||
		cfg.Output == "" {
		return errors.New("steady: all split paths and output are required")
	}
	if cfg.Bucket < 1 || cfg.Bucket > 1_000_000 ||
		cfg.Dimension < 1 || cfg.Dimension > 4096 || cfg.MinN < 1 ||
		cfg.MaxN < cfg.MinN || cfg.MaxN > 16 || cfg.Epochs < 1 ||
		cfg.Epochs > 10000 || cfg.LearningRate <= 0 || cfg.LearningRate > 10 ||
		cfg.L2 < 0 || cfg.L2 > 10 || cfg.Alpha <= 0 || cfg.Alpha >= 1 ||
		cfg.PositiveWeightScale <= 0 {
		return errors.New("steady: invalid v4 training configuration")
	}
	labels := []string{"short", "medium", "long"}
	training, err := parseExamples(cfg.TrainInput, labels)
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
	if len(training) == 0 || len(probability) == 0 || len(conformal) == 0 || len(development) == 0 {
		return errors.New("steady: every frozen split must be non-empty")
	}
	positiveWeights := v4PositiveWeights(training, cfg.PositiveWeightScale)
	stride := cfg.Dimension + temporalFeatureCount
	table := deterministicValues(cfg.Bucket*cfg.Dimension, cfg.Seed^0x91e10da5)
	weights := deterministicValues(2*stride, cfg.Seed^0xd1b54a32)
	bias := make([]float32, 2)
	indices := make([]int, 0, 8192)
	temporal := make([]float32, temporalFeatureCount)
	hidden := make([]float32, cfg.Dimension)
	gradientHidden := make([]float32, cfg.Dimension)
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		for _, ex := range training {
			indices = sparseBuckets(ex.text, cfg.Bucket, cfg.MinN, cfg.MaxN, indices[:0])
			temporalValues(ex.text, temporal)
			clear(hidden)
			scale := float32(0)
			if len(indices) > 0 {
				scale = float32(1 / float64(len(indices)))
				for _, index := range indices {
					base := index * cfg.Dimension
					for feature := 0; feature < cfg.Dimension; feature++ {
						hidden[feature] += table[base+feature]
					}
				}
				for feature := range hidden {
					hidden[feature] *= scale
				}
			}
			clear(gradientHidden)
			for head, labelIndex := range []int{0, 2} {
				base := head * stride
				logit := bias[head]
				for feature, value := range hidden {
					logit += weights[base+feature] * value
				}
				for feature, value := range temporal {
					logit += weights[base+cfg.Dimension+feature] * value
				}
				target := float32(0)
				sampleWeight := float32(1)
				if ex.label == labelIndex {
					target = 1
					sampleWeight = positiveWeights[head]
				}
				gradient := (sigmoid(logit, 1) - target) * sampleWeight
				bias[head] -= cfg.LearningRate * gradient
				for feature, value := range hidden {
					position := base + feature
					gradientHidden[feature] += gradient * weights[position]
					weights[position] -= cfg.LearningRate *
						(gradient*value + cfg.L2*weights[position])
				}
				for feature, value := range temporal {
					position := base + cfg.Dimension + feature
					weights[position] -= cfg.LearningRate *
						(gradient*value + cfg.L2*weights[position])
				}
			}
			for _, index := range indices {
				base := index * cfg.Dimension
				for feature := 0; feature < cfg.Dimension; feature++ {
					position := base + feature
					table[position] -= cfg.LearningRate *
						(gradientHidden[feature]*scale + cfg.L2*table[position])
				}
			}
		}
	}
	model := &Model{
		table:            table,
		weights:          weights,
		bias:             bias,
		temperatures:     []float32{1, 1},
		quantiles:        make([]float32, 4),
		thresholds:       []float32{1, 1},
		bucket:           cfg.Bucket,
		dim:              cfg.Dimension,
		minN:             cfg.MinN,
		maxN:             cfg.MaxN,
		wordMinN:         1,
		wordMaxN:         2,
		temporalFeatures: temporalFeatureCount,
		metadata: Metadata{
			ArtifactFormat:       int(modelVersion),
			ModelID:              "settings-v2",
			Task:                 "video-duration-selection",
			Labels:               labels,
			PolicyCompatibility:  ProfileQuotaSafeV2,
			MinN:                 cfg.MinN,
			MaxN:                 cfg.MaxN,
			Bucket:               cfg.Bucket,
			Dimension:            cfg.Dimension,
			Epochs:               cfg.Epochs,
			LearningRate:         float64(cfg.LearningRate),
			L2:                   float64(cfg.L2),
			Alpha:                cfg.Alpha,
			Seed:                 cfg.Seed,
			SourceManifestSHA256: cfg.SourceManifestSHA256,
			TrainingCodeCommit:   cfg.TrainingCodeCommit,
			ModelFamily:          "dual-head-embedding-v1",
			FeatureSchema:        "hashed-char-word-embedding+temporal-v1",
			Heads:                []string{"short", "long"},
			WordMinN:             1,
			WordMaxN:             2,
			TemporalFeatures:     temporalFeatureCount,
			PositiveClassWeights: []float64{
				float64(positiveWeights[0]), float64(positiveWeights[1]),
			},
		},
	}
	if err := validateMetadata(model.metadata); err != nil {
		return err
	}
	model.temperatures = fitV4Temperatures(model, probability)
	model.quantiles = fitV4Quantiles(model, conformal, cfg.Alpha)
	model.thresholds = fitV4Thresholds(model, development)
	artifact, err := marshalModel(model)
	if err != nil {
		return err
	}
	if len(artifact) > maxReleaseArtifactSize {
		return fmt.Errorf("steady: v4 artifact exceeds %d-byte release limit", maxReleaseArtifactSize)
	}
	if err := os.WriteFile(cfg.Output, artifact, 0o644); err != nil {
		return fmt.Errorf("steady: write artifact: %w", err)
	}
	return nil
}

func v4PositiveWeights(examples []example, multiplier float32) []float32 {
	out := make([]float32, 2)
	for head, labelIndex := range []int{0, 2} {
		positive := 0
		for _, ex := range examples {
			if ex.label == labelIndex {
				positive++
			}
		}
		negative := len(examples) - positive
		if positive == 0 || negative == 0 {
			out[head] = 1
		} else {
			out[head] = min(float32(20), float32(negative)/float32(positive)*multiplier)
		}
	}
	return out
}

func v4Probabilities(model *Model, examples []example, temperatures []float32) [][]float32 {
	out := make([][]float32, len(examples))
	indices := make([]int, 0, 8192)
	temporal := make([]float32, temporalFeatureCount)
	hidden := make([]float32, model.dim)
	for row, ex := range examples {
		out[row] = make([]float32, 2)
		indices = v4Predict(
			ex.text, model.table, model.weights, model.bias, temperatures,
			model.bucket, model.dim, model.temporalFeatures, model.minN, model.maxN,
			hidden,
			out[row], indices[:0], temporal,
		)
	}
	return out
}

func fitV4Temperatures(model *Model, examples []example) []float32 {
	out := []float32{1, 1}
	base := v4Probabilities(model, examples, out)
	logits := make([][]float32, len(base))
	for row := range base {
		logits[row] = make([]float32, 2)
		for head, probability := range base[row] {
			p := min(float64(1-1e-7), max(float64(1e-7), float64(probability)))
			logits[row][head] = float32(math.Log(p / (1 - p)))
		}
	}
	for head, labelIndex := range []int{0, 2} {
		bestLoss := math.Inf(1)
		for step := 1; step <= 200; step++ {
			temperature := float32(step) / 20
			loss := 0.0
			for row, ex := range examples {
				p := float64(sigmoid(logits[row][head], temperature))
				if ex.label == labelIndex {
					loss -= math.Log(math.Max(p, 1e-12))
				} else {
					loss -= math.Log(math.Max(1-p, 1e-12))
				}
			}
			if loss < bestLoss {
				bestLoss, out[head] = loss, temperature
			}
		}
	}
	return out
}

func fitV4Quantiles(model *Model, examples []example, alpha float64) []float32 {
	probabilities := v4Probabilities(model, examples, model.temperatures)
	scores := make([][]float32, 4)
	for row, ex := range examples {
		for head, labelIndex := range []int{0, 2} {
			p := probabilities[row][head]
			if ex.label == labelIndex {
				scores[head*2] = append(scores[head*2], 1-p)
			} else {
				scores[head*2+1] = append(scores[head*2+1], p)
			}
		}
	}
	out := make([]float32, 4)
	for index := range scores {
		out[index] = conformalQuantile(scores[index], alpha)
	}
	return out
}

func fitV4Thresholds(model *Model, examples []example) []float32 {
	probabilities := v4Probabilities(model, examples, model.temperatures)
	out := []float32{1, 1}
	for head, labelIndex := range []int{0, 2} {
		minimum := []float64{0.95, 0.98}[head]
		actualPositives := 0
		for _, ex := range examples {
			if ex.label == labelIndex {
				actualPositives++
			}
		}
		minimumAccepted := max(5, (actualPositives+3)/4)
		for step := 160; step <= 199; step++ {
			threshold := float32(step) * 0.005
			accepted, correct := 0, 0
			for row, ex := range examples {
				p := probabilities[row][head]
				positiveIncluded := 1-p <= model.quantiles[head*2]
				negativeIncluded := p <= model.quantiles[head*2+1]
				cueEligible := head != 1 || longCueEligible(ex.text)
				if cueEligible && positiveIncluded && !negativeIncluded && p >= threshold {
					accepted++
					if ex.label == labelIndex {
						correct++
					}
				}
			}
			if accepted >= minimumAccepted &&
				float64(correct)/float64(accepted) >= minimum {
				out[head] = threshold
				break
			}
		}
	}
	return out
}
