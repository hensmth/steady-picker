package steady

import (
	"encoding/json"
	"testing"
)

func FuzzRequestJSON(f *testing.F) {
	f.Add([]byte(`{"prompt":"a cat walks","mode":"text-to-video"}`))
	f.Add([]byte(`{"prompt":"not 2 seconds then 6 seconds","mode":"text-to-video"}`))
	f.Fuzz(func(t *testing.T, input []byte) {
		var request PickRequest
		if json.Unmarshal(input, &request) == nil {
			_, _ = PickSettings(nil, QuotaSafeProfile(), request)
			_ = affirmativeDurationCues(request.Prompt)
		}
	})
}
