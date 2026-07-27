# Contributing to SteadyPicker

## Issues

- Search existing issues before opening a new one
- Include a minimal reproduction case
- For bug reports, include Go version and platform

## Pull Requests

- Keep PRs focused on a single change
- Run `go test ./...`, `go test -race ./...`, and `go vet ./...` before
  submitting
- Every exported function must have a Go doc comment starting with the name
- Cyclomatic complexity must not exceed 10 per function
- Keep prompts on stdin; do not add prompts to command-line arguments or logs.
- Keep corpora, teacher output, credentials, generated media, and experimental
  models out of Git. Changes to the committed default model require an updated
  model card, checksum test, and validation evidence.
- Add tests for new policy or model-format behavior.

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Go doc comments on all exported types, functions, methods, and constants
- Production inference must remain network-free.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
