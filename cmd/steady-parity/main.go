// Command steady-parity exposes raw classifier probabilities for parity checks.
// It is a development utility and is not included in release archives.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	steady "github.com/hensmth/steady-picker"
)

type request struct {
	Prompt string `json:"prompt"`
}

type response struct {
	Probabilities []float32 `json:"probabilities"`
}

func main() {
	modelPath := flag.String("model", "", "v5 artifact to inspect")
	flag.Parse()
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}
	model, err := steady.Load(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4096), 32<<10)
	encoder := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var input request
		if err := json.Unmarshal(scanner.Bytes(), &input); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		debug := model.ClassifyDebug(input.Prompt)
		if debug.IsEmpty {
			fmt.Fprintln(os.Stderr, "empty classification")
			os.Exit(1)
		}
		if err := encoder.Encode(response{Probabilities: debug.Probabilities}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
