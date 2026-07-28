// Package steady provides local, quota-aware video setting selection in pure Go.
//
// Models use strict, provenance-carrying v3 artifacts and are safe for concurrent
// use. Duration is learned; aspect ratio and resolution remain deterministic.
//
//	m, err := steady.LoadDefault()
//	if err != nil { /* handle error */ }
//	result, err := steady.PickSettings(m, steady.QuotaSafeProfile(), steady.PickRequest{
//		Prompt: "a flower transforms through three ordered stages",
//		Mode: steady.TextToVideo,
//	})
package steady
