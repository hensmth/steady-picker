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
- Keep credentials, private prompts, generated media, checkpoints, caches, and
  experimental models out of Git. The governed sanitized corpus, label votes,
  split membership, default model, manifests, and evaluation report are public
  release inputs and require matching cards, digests, and validation evidence.
- Add tests for new policy or model-format behavior.

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Go doc comments on all exported types, functions, methods, and constants
- Production inference must remain network-free.

## License

Source contributions are MIT. Contributions to the published v2 corpus or
trained model are CC-BY-4.0.
