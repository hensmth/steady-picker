// Package steady provides local, quota-aware video setting selection in pure Go.
//
// The default trained model is embedded. Custom models can be memory-mapped
// from disk or loaded from bytes.
// Classification runs the byteSteady encoder, OVA logistic regression,
// Platt scaling, and conformal prediction in a single sub-millisecond pass
// with zero heap allocations.
//
// Quick start:
//
//	m, _ := steady.LoadDefault()
//	defer m.Close()
//	result := steady.PickSettings(m, steady.PickRequest{
//		Prompt: "a flower transforming in a timelapse",
//		Mode:   "text-to-video",
//	}, "v1")
//
// Training:
//
//	go run ./cmd/steady-picker train --input data.txt --output model.bin
//
// Input format: __label__d2 Text here
package steady
