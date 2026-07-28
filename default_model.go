package steady

import _ "embed"

// DefaultModelVersion identifies the model embedded in released binaries.
const DefaultModelVersion = "settings-v2"

//go:embed models/settings-v2.bin
var defaultModelArtifact []byte

// LoadDefault loads the immutable trained settings model embedded in this package.
func LoadDefault() (*Model, error) {
	return LoadBytes(defaultModelArtifact)
}
