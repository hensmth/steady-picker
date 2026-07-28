# Artifact format v5

Format v5 is SteadyPicker's strict, self-contained semantic-model format. The
loader continues to accept formats v3 and v4. Format v2 remains rejected because
its label order is ambiguous.

All integers are little-endian. The 32-byte header is:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | `uint32` | magic `0x53544459` |
| 4 | `uint32` | format `5` |
| 8 | `uint32` | canonical JSON metadata length |
| 12 | `uint32` | reserved, zero |
| 16 | `uint64` | payload length |
| 24 | `uint64` | reserved, zero |

The metadata fixes the architecture at two Transformer layers, hidden size 128,
four attention heads, FFN size 512, an 8,192-token WordPiece vocabulary, and a
96-token maximum. It also carries semantic label/head order, calibration,
policy compatibility, teacher/training provenance, source-manifest digest, and
training-code commit. The artifact SHA-256 is derived by the loader and is never
accepted from metadata.

The payload is a fixed sequence:

1. token embeddings, position embeddings, and embedding layer normalization;
2. query, key, value, attention output, attention layer normalization, two FFN
   matrices, and output layer normalization for each layer;
3. two duration heads and six temporal auxiliary heads;
4. two temperatures, four class-conditional conformal quantiles, and two frozen
   probability thresholds.

Every matrix stores row-major signed INT8 weights followed by one float32 scale
per output row and, where applicable, one float32 bias per output row.
Layer-normalization weights and calibration values remain float32.

The loader validates the full canonical metadata, vocabulary uniqueness and
special-token order, every fixed dimension, exact payload size, every numeric
value, and the 64 MiB global artifact ceiling before constructing a model.
Release artifacts have the stricter 32 MiB ceiling.
