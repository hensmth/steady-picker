package steady

import _ "embed"

// DefaultModelVersion identifies the embedded artifact interface.
const DefaultModelVersion = "settings-v2"

//go:embed models/settings-v2.bin
var defaultModelArtifact []byte

// LoadDefault loads the embedded immutable artifact. Development builds may
// contain the release-blocked bootstrap artifact reported by the health command.
func LoadDefault() (*Model, error) {
	return LoadBytes(defaultModelArtifact)
}
