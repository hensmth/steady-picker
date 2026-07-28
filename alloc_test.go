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
