package steady_test

import (
	"fmt"
	"os"
	"strings"

	steady "github.com/hensmth/steady-picker"
)

// Example_classify demonstrates loading a model and classifying text.
func Example_classify() {
	lines := []string{
		"__label__spam buy cheap watches now",
		"__label__urgent server down in production",
		"__label__question how do I reset my password",
		"__label__update deployed v2.3.1 to staging",
		"__label__complaint the checkout page is broken",
		"__label__praise the new dashboard looks amazing",
	}
	input := strings.Repeat(strings.Join(lines, "\n")+"\n", 30)
	tmpDir, _ := os.MkdirTemp("", "steady_example_*")
	defer os.RemoveAll(tmpDir)
	inputPath := tmpDir + "/train.txt"
	outputPath := tmpDir + "/model.bin"
	os.WriteFile(inputPath, []byte(input), 0644)

	cfg := steady.DefaultTrainConfig()
	cfg.Input = inputPath
	cfg.Output = outputPath
	cfg.Bucket = 500
	cfg.Dim = 8
	cfg.Epochs = 10
	cfg.LR = 0.2
	cfg.Seed = 42
	cfg.LabelNames = []string{"spam", "urgent", "question", "update", "complaint", "praise"}
	if err := steady.Train(cfg); err != nil {
		panic(err)
	}

	m, err := steady.Load(outputPath)
	if err != nil {
		panic(err)
	}
	defer m.Close()
	m.SetLabelNames(cfg.LabelNames)

	result := m.Classify("buy cheap watches now")
	if result.IsEmpty() {
		fmt.Println("noise")
	} else {
		fmt.Println(result.Kinds[0])
	}
	// Output:
	// spam
}
