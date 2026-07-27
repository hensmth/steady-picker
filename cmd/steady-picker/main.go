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
		fatal("usage: steady-picker <predict|train|evaluate|version>")
	}
	switch os.Args[1] {
	case "predict":
		predict(os.Args[2:])
	case "train":
		train(os.Args[2:])
	case "evaluate":
		evaluate(os.Args[2:])
	case "version":
		fmt.Println(buildVersion())
	default:
		fatal("unknown command %q", os.Args[1])
	}
}

func buildVersion() string {
	if version != "dev" {
		return version
	}
	if info, ok := debug.ReadBuildInfo(); ok &&
		info.Main.Version != "" &&
		info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return version
}

type metrics struct {
	Examples         int                `json:"examples"`
	Correct          int                `json:"correct"`
	Accuracy         float64            `json:"accuracy"`
	MacroF1          float64            `json:"macro_f1"`
	ByLabel          map[string]float64 `json:"f1_by_label"`
	Accepted         int                `json:"accepted"`
	AcceptedCorrect  int                `json:"accepted_correct"`
	AcceptedAccuracy float64            `json:"accepted_accuracy"`
	Coverage         float64            `json:"coverage"`
}

func evaluate(args []string) {
	flags := flag.NewFlagSet("evaluate", flag.ExitOnError)
	modelPath := flags.String("model", "", "trained model path")
	testPath := flags.String("test", "", "labelled test file")
	_ = flags.Parse(args)
	if *modelPath == "" || *testPath == "" {
		fatal("evaluate requires --model and --test")
	}
	model, err := steady.Load(*modelPath)
	if err != nil {
		fatal("load model: %v", err)
	}
	defer model.Close()
	model.SetLabelNames(steady.PresetLabels)
	file, err := os.Open(*testPath)
	if err != nil {
		fatal("open test file: %v", err)
	}
	defer file.Close()

	index := map[string]int{"d2": 0, "d4": 1, "d6": 2}
	confusion := make([][]int, 3)
	for i := range confusion {
		confusion[i] = make([]int, 3)
	}
	result := metrics{ByLabel: map[string]float64{}}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 16*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "__label__") {
			continue
		}
		space := strings.IndexByte(line, ' ')
		if space < 0 {
			continue
		}
		label := strings.TrimPrefix(line[:space], "__label__")
		expected, ok := index[label]
		if !ok {
			continue
		}
		debug := model.ClassifyDebug(strings.TrimSpace(line[space+1:]))
		predicted := 0
		for i := 1; i < len(debug.Calibrated); i++ {
			if debug.Calibrated[i] > debug.Calibrated[predicted] {
				predicted = i
			}
		}
		confusion[expected][predicted]++
		result.Examples++
		if expected == predicted {
			result.Correct++
		}
		picked := steady.PickSettings(
			model,
			steady.PickRequest{Prompt: strings.TrimSpace(line[space+1:])},
			"evaluation",
		)
		if picked.Source == "model" {
			result.Accepted++
			if picked.Duration == 2*(expected+1) {
				result.AcceptedCorrect++
			}
		}
	}
	if err := scanner.Err(); err != nil {
		fatal("read test file: %v", err)
	}
	if result.Examples == 0 {
		fatal("test file contained no examples")
	}
	result.Accuracy = float64(result.Correct) / float64(result.Examples)
	result.Coverage = float64(result.Accepted) / float64(result.Examples)
	if result.Accepted > 0 {
		result.AcceptedAccuracy = float64(result.AcceptedCorrect) / float64(result.Accepted)
	}
	for label, i := range index {
		tp := confusion[i][i]
		var actual, predicted int
		for j := range confusion {
			actual += confusion[i][j]
			predicted += confusion[j][i]
		}
		denominator := actual + predicted
		if denominator > 0 {
			result.ByLabel[label] = float64(2*tp) / float64(denominator)
		}
		result.MacroF1 += result.ByLabel[label] / float64(len(index))
	}
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fatal("encode metrics: %v", err)
	}
}

func predict(args []string) {
	flags := flag.NewFlagSet("predict", flag.ExitOnError)
	modelPath := flags.String("model", "", "custom model path (default: embedded settings-v1)")
	modelVersion := flags.String(
		"model-version",
		steady.DefaultModelVersion,
		"model version in output",
	)
	_ = flags.Parse(args)
	if flags.NArg() != 0 {
		fatal("predict accepts one JSON object on stdin; positional arguments are not allowed")
	}

	var (
		model *steady.Model
		err   error
	)
	if strings.TrimSpace(*modelPath) != "" {
		model, err = steady.Load(*modelPath)
	} else {
		model, err = steady.LoadDefault()
	}
	if err != nil {
		fatal("load model: %v", err)
	}
	defer model.Close()

	request, err := decodePickRequest(os.Stdin)
	if err != nil {
		fatal("read request: %v", err)
	}
	result := steady.PickSettings(model, request, *modelVersion)
	if err := json.NewEncoder(os.Stdout).Encode(result); err != nil {
		fatal("encode result: %v", err)
	}
}

func decodePickRequest(reader io.Reader) (steady.PickRequest, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), 16*1024)
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return steady.PickRequest{}, err
		}
		return steady.PickRequest{}, fmt.Errorf("empty stdin")
	}
	var request steady.PickRequest
	if err := json.Unmarshal(scanner.Bytes(), &request); err != nil {
		return steady.PickRequest{}, fmt.Errorf("decode JSON: %w", err)
	}
	if scanner.Scan() {
		return steady.PickRequest{}, fmt.Errorf("expected exactly one JSON object")
	}
	if err := scanner.Err(); err != nil {
		return steady.PickRequest{}, err
	}
	if strings.TrimSpace(request.Prompt) == "" {
		return steady.PickRequest{}, fmt.Errorf("prompt is required")
	}
	if request.Mode != "text-to-video" && request.Mode != "image-to-video" {
		return steady.PickRequest{}, fmt.Errorf(
			"mode must be text-to-video or image-to-video",
		)
	}
	if request.ImageAspectRatio < 0 {
		return steady.PickRequest{}, fmt.Errorf("image_aspect_ratio cannot be negative")
	}
	return request, nil
}

func train(args []string) {
	flags := flag.NewFlagSet("train", flag.ExitOnError)
	cfg := steady.DefaultTrainConfig()
	input := flags.String("input", "", "training data path")
	output := flags.String("output", "model.bin", "model output path")
	bucket := flags.Int("bucket", 20000, "embedding rows")
	dim := flags.Int("dim", 32, "embedding dimension")
	epochs := flags.Int("epochs", 150, "training epochs")
	learningRate := flags.Float64("lr", 0.5, "learning rate")
	_ = flags.Parse(args)

	cfg.Input = *input
	cfg.Output = *output
	cfg.Bucket = *bucket
	cfg.Dim = *dim
	cfg.Epochs = *epochs
	cfg.LR = float32(*learningRate)
	cfg.NumGoroutines = 1
	cfg.LabelNames = append([]string(nil), steady.PresetLabels...)
	if cfg.Input == "" {
		fatal("train requires --input")
	}
	if err := steady.Train(cfg); err != nil {
		fatal("train: %v", err)
	}
	fmt.Println(cfg.Output)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "steady-picker: "+format+"\n", args...)
	os.Exit(1)
}
