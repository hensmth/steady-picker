package steady

import _ "embed"

//go:embed THIRD_PARTY_NOTICES.md
var notices string

// Licenses returns the attribution and third-party notices embedded in the
// standalone library and CLI.
func Licenses() string {
	return notices
}
