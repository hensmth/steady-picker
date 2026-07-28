package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strings"

	steady "github.com/hensmth/steady-picker"
)

var version = "dev"

func main() {
	if len(os.Args) < 2 {
		fatal("usage: steady-picker <predict|inspect-model|health|licenses|train|version>")
	}
	switch os.Args[1] {
	case "predict":
		predict(os.Args[2:])
	case "inspect-model":
		inspectModel(os.Args[2:])
	case "health":
		health(os.Args[2:])
	case "licenses":
		licenses()
	case "train":
		train(os.Args[2:])
	case "version":
		fmt.Println(buildVersion())
	default:
		fatal("unknown command %q", os.Args[1])
	}
}

func loadModel(path string) (*steady.Model, error) {
	if strings.TrimSpace(path) == "" {
		return steady.LoadDefault()
	}
	return steady.Load(path)
}

func loadProfile(name, path string) (steady.Profile, error) {
	if path == "" {
		if name != steady.ProfileQuotaSafeV2 {
			return steady.Profile{}, fmt.Errorf("unknown embedded profile %q", name)
		}
		return steady.QuotaSafeProfile(), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return steady.Profile{}, err
	}
	var config steady.ProfileConfig
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return steady.Profile{}, err
	}
	return steady.NewProfile(config)
}

func predict(args []string) {
	flags := flag.NewFlagSet("predict", flag.ExitOnError)
	modelPath := flags.String("model", "", "optional custom v3/v4/v5 artifact")
	profileName := flags.String("profile", steady.ProfileQuotaSafeV2, "embedded profile")
	profilePath := flags.String("profile-file", "", "custom profile JSON")
	_ = flags.Parse(args)
	if flags.NArg() != 0 {
		fatal("predict accepts newline-delimited JSON exclusively through stdin")
	}
	model, err := loadModel(*modelPath)
	if err != nil {
		fatal("load model: %v", err)
	}
	profile, err := loadProfile(*profileName, *profilePath)
	if err != nil {
		fatal("load profile: %v", err)
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 32<<10)
	encoder := json.NewEncoder(os.Stdout)
	rows := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var request steady.PickRequest
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&request); err != nil {
			fatal("decode request at line %d: %v", rows+1, err)
		}
		result, err := steady.PickSettings(model, profile, request)
		if err != nil {
			fatal("pick settings at line %d: %v", rows+1, err)
		}
		if err := encoder.Encode(result); err != nil {
			fatal("encode result: %v", err)
		}
		rows++
	}
	if err := scanner.Err(); err != nil {
		fatal("read stdin: %v", err)
	}
	if rows == 0 {
		fatal("stdin contained no JSON objects")
	}
}

func inspectModel(args []string) {
	flags := flag.NewFlagSet("inspect-model", flag.ExitOnError)
	path := flags.String("model", "", "optional custom v3/v4/v5 artifact")
	_ = flags.Parse(args)
	model, err := loadModel(*path)
	if err != nil {
		fatal("load model: %v", err)
	}
	if err := json.NewEncoder(os.Stdout).Encode(model.Metadata()); err != nil {
		fatal("encode metadata: %v", err)
	}
}

func health(args []string) {
	flags := flag.NewFlagSet("health", flag.ExitOnError)
	path := flags.String("model", "", "optional custom v3/v4/v5 artifact")
	_ = flags.Parse(args)
	model, err := loadModel(*path)
	if err != nil {
		fatal("unhealthy: %v", err)
	}
	_, err = steady.PickSettings(model, steady.QuotaSafeProfile(), steady.PickRequest{
		Prompt: "health check", Mode: steady.TextToVideo,
	})
	if err != nil {
		fatal("unhealthy: %v", err)
	}
	metadata := model.Metadata()
	status := "ok"
	ready := true
	if metadata.TrainingCodeCommit == "bootstrap-not-for-release" {
		status = "bootstrap"
		ready = false
	}
	fmt.Printf(
		"{\"status\":%q,\"ready\":%t,\"model_version\":%q,"+
			"\"artifact_format\":%d,\"policy_version\":%q,"+
			"\"artifact_sha256\":%q}\n",
		status, ready, metadata.ModelID, metadata.ArtifactFormat,
		steady.ProfileQuotaSafeV2, metadata.ArtifactSHA256,
	)
}

func licenses() {
	io.WriteString(os.Stdout, steady.Licenses())
}

func train(args []string) {
	flags := flag.NewFlagSet("train", flag.ExitOnError)
	cfg := steady.DefaultTrainConfig()
	flags.StringVar(&cfg.TrainInput, "train", "", "frozen training split")
	flags.StringVar(&cfg.ProbabilityCalibrationInput, "probability-calibration", "", "probability calibration split")
	flags.StringVar(&cfg.ConformalCalibrationInput, "conformal-calibration", "", "conformal calibration split")
	flags.StringVar(&cfg.ThresholdDevelopmentInput, "threshold-development", "", "policy threshold split")
	flags.StringVar(&cfg.Output, "output", "settings-v2.bin", "artifact output")
	flags.IntVar(&cfg.Bucket, "bucket", cfg.Bucket, "hash buckets")
	flags.IntVar(&cfg.Dimension, "dimension", cfg.Dimension, "embedding dimension")
	flags.IntVar(&cfg.MinN, "min-ngram", cfg.MinN, "minimum byte n-gram")
	flags.IntVar(&cfg.MaxN, "max-ngram", cfg.MaxN, "maximum byte n-gram")
	flags.IntVar(&cfg.Epochs, "epochs", cfg.Epochs, "epochs")
	lr := flags.Float64("learning-rate", float64(cfg.LearningRate), "learning rate")
	l2 := flags.Float64("l2", float64(cfg.L2), "L2 regularization")
	positiveWeightScale := flags.Float64(
		"positive-weight-scale",
		float64(cfg.PositiveWeightScale),
		"minority positive-class weight multiplier",
	)
	flags.StringVar(&cfg.SourceManifestSHA256, "source-manifest-sha256", "", "source manifest SHA-256")
	flags.StringVar(&cfg.TrainingCodeCommit, "training-code-commit", "", "training code commit")
	_ = flags.Parse(args)
	cfg.LearningRate, cfg.L2 = float32(*lr), float32(*l2)
	cfg.PositiveWeightScale = float32(*positiveWeightScale)
	if err := steady.Train(cfg); err != nil {
		fatal("train: %v", err)
	}
	fmt.Println(cfg.Output)
}

func buildVersion() string {
	if version != "dev" {
		return version
	}
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return version
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "steady-picker: "+format+"\n", args...)
	os.Exit(1)
}
