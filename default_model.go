package steady

import _ "embed"

// DefaultModelVersion identifies the model embedded in released binaries.
const DefaultModelVersion = "v1"

//go:embed models/settings-v1.bin
var defaultModelArtifact []byte

// LoadDefault loads the trained settings model embedded in this package.
// The caller must call Close to release its resources.
func LoadDefault() (*Model, error) {
	return LoadBytes(defaultModelArtifact)
}
