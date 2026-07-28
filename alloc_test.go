//go:build !race

package steady

import "testing"

func TestSteadyStatePickSettingsHasZeroAllocations(t *testing.T) {
	model, _ := testModel(t)
	profile := QuotaSafeProfile()
	request := PickRequest{Prompt: "a person walks naturally", Mode: TextToVideo}
	allocations := testing.AllocsPerRun(1000, func() {
		if _, err := PickSettings(model, profile, request); err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("steady-state allocations = %.2f, want 0", allocations)
	}
}

func TestV4SteadyStatePickSettingsHasZeroAllocations(t *testing.T) {
	model, _ := testV4Model(t)
	profile := QuotaSafeProfile()
	request := PickRequest{Prompt: "one quick wink", Mode: TextToVideo}
	_, _ = PickSettings(model, profile, request)
	allocations := testing.AllocsPerRun(1000, func() {
		if _, err := PickSettings(model, profile, request); err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("v4 steady-state allocations = %.2f, want 0", allocations)
	}
}

func TestV5SteadyStatePickSettingsHasZeroAllocations(t *testing.T) {
	model, err := LoadBytes(semanticTestArtifact(t, []float32{8, -8}))
	if err != nil {
		t.Fatal(err)
	}
	profile := QuotaSafeProfile()
	request := PickRequest{Prompt: "one quick wink", Mode: TextToVideo}
	_, _ = PickSettings(model, profile, request)
	allocations := testing.AllocsPerRun(100, func() {
		if _, err := PickSettings(model, profile, request); err != nil {
			panic(err)
		}
	})
	if allocations != 0 {
		t.Fatalf("v5 steady-state allocations = %.2f, want 0", allocations)
	}
}
