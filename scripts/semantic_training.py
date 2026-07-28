"""Deterministic semantic student, tokenizer, calibration, and v5 export."""

from __future__ import annotations

import collections
import copy
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MAGIC = 0x53544459
FORMAT = 5
VOCAB_SIZE = 8192
MAX_TOKENS = 96
HIDDEN = 128
ATTENTION_HEADS = 4
INTERMEDIATE = 512
LAYERS = 2
AUXILIARY_HEADS = [
    "beats_1",
    "beats_2",
    "beats_3",
    "ordered",
    "transformation",
    "dependent_actions",
]
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOKEN = re.compile(r"[a-z0-9_']+|[^\\s\\w]", re.ASCII)


def build_vocabulary(texts: Iterable[str]) -> list[str]:
    words: collections.Counter[str] = collections.Counter()
    pieces: collections.Counter[str] = collections.Counter()
    punctuation: collections.Counter[str] = collections.Counter()
    for text in texts:
        normalized = "".join(
            character.lower() if ord(character) < 128 else " "
            for character in text
        )
        for token in TOKEN.findall(normalized):
            if re.fullmatch(r"[a-z0-9_']+", token):
                words[token] += 1
                for length in range(1, min(6, len(token) + 1)):
                    pieces[token[:length]] += 1
                for start in range(1, len(token)):
                    for length in range(1, min(6, len(token) - start + 1)):
                        pieces["##" + token[start : start + length]] += 1
            else:
                punctuation[token] += 1
    vocabulary = list(SPECIAL)
    seen = set(vocabulary)

    def extend(candidates: Iterable[str], limit: int | None = None) -> None:
        for token in candidates:
            if token in seen or not token or len(token.encode("utf-8")) > 128:
                continue
            vocabulary.append(token)
            seen.add(token)
            if len(vocabulary) == VOCAB_SIZE or (
                limit is not None and len(vocabulary) >= limit
            ):
                return

    ranked_punctuation = [
        token for token, _ in sorted(punctuation.items(), key=lambda item: (-item[1], item[0]))
    ]
    ranked_words = [
        token for token, _ in sorted(words.items(), key=lambda item: (-item[1], item[0]))
    ]
    ranked_pieces = [
        token for token, _ in sorted(pieces.items(), key=lambda item: (-item[1], item[0]))
    ]
    extend(ranked_punctuation)
    extend(ranked_words, 6_000)
    extend(ranked_pieces)
    counter = 0
    while len(vocabulary) < VOCAB_SIZE:
        token = f"[unused-{counter:05d}]"
        counter += 1
        if token not in seen:
            vocabulary.append(token)
            seen.add(token)
    if len(vocabulary) != VOCAB_SIZE or len(seen) != VOCAB_SIZE:
        raise RuntimeError("failed to build an exact unique WordPiece vocabulary")
    return vocabulary


class WordPiece:
    def __init__(self, vocabulary: list[str]):
        if len(vocabulary) != VOCAB_SIZE or vocabulary[:5] != SPECIAL:
            raise ValueError("invalid semantic vocabulary")
        self.vocabulary = vocabulary
        self.ids = {token: index for index, token in enumerate(vocabulary)}

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        lowered = bytes(
            value + 32 if ord("A") <= value <= ord("Z") else value
            for value in raw
        )
        output = [2]
        start = 0
        while start < len(lowered) and len(output) < MAX_TOKENS - 1:
            while start < len(lowered) and lowered[start] in b" \t\r\n":
                start += 1
            if start == len(lowered):
                break
            end = start + 1
            if _ascii_word(lowered[start]):
                while end < len(lowered) and _ascii_word(lowered[end]):
                    end += 1
            elif lowered[start] >= 128:
                while end < len(lowered) and lowered[end] >= 128:
                    end += 1
            output.extend(
                self._word(lowered[start:end], MAX_TOKENS - 1 - len(output))
            )
            start = end
        return output[: MAX_TOKENS - 1] + [3]

    def _word(self, word: bytes, remaining: int) -> list[int]:
        if not word or len(word) > 100 or any(value >= 128 for value in word):
            return [1] if remaining else []
        output: list[int] = []
        start = 0
        while start < len(word) and len(output) < remaining:
            found: tuple[int, int] | None = None
            for end in range(len(word), start, -1):
                value = word[start:end].decode("ascii")
                token = value if start == 0 else "##" + value
                token_id = self.ids.get(token)
                if token_id is not None:
                    found = token_id, end
                    break
            if found is None:
                return [1]
            output.append(found[0])
            start = found[1]
        return output

    def batch(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch_ids([self.encode(text) for text in texts], device)

    def batch_ids(
        self, encoded: list[list[int]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(row) for row in encoded)
        ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool, device=device)
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[index, : len(row)] = True
        return ids, mask

    def cache(self, rows: list[dict]) -> None:
        for row in rows:
            row["_semantic_token_ids"] = self.encode(row["prompt"])

    def batch_rows(
        self, rows: list[dict], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch_ids(
            [
                row.get("_semantic_token_ids") or self.encode(row["prompt"])
                for row in rows
            ],
            device,
        )


def _ascii_word(value: int) -> bool:
    return (
        ord("a") <= value <= ord("z")
        or ord("0") <= value <= ord("9")
        or value in (ord("_"), ord("'"))
    )


class SemanticLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(HIDDEN, HIDDEN)
        self.key = nn.Linear(HIDDEN, HIDDEN)
        self.value = nn.Linear(HIDDEN, HIDDEN)
        self.attention_output = nn.Linear(HIDDEN, HIDDEN)
        self.attention_norm = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.feed_forward_in = nn.Linear(HIDDEN, INTERMEDIATE)
        self.feed_forward_out = nn.Linear(INTERMEDIATE, HIDDEN)
        self.output_norm = nn.LayerNorm(HIDDEN, eps=1e-5)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        head_width = HIDDEN // ATTENTION_HEADS

        def heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.reshape(
                batch, tokens, ATTENTION_HEADS, head_width
            ).transpose(1, 2)

        query = heads(self.query(values))
        key = heads(self.key(values))
        value = heads(self.value(values))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_width)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, tokens, HIDDEN
        )
        values = self.attention_norm(values + self.attention_output(context))
        hidden = F.gelu(self.feed_forward_in(values), approximate="tanh")
        return self.output_norm(values + self.feed_forward_out(hidden))


class SemanticStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.position_embeddings = nn.Embedding(MAX_TOKENS, HIDDEN)
        self.embedding_norm = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.layers = nn.ModuleList(SemanticLayer() for _ in range(LAYERS))
        self.duration_heads = nn.Linear(HIDDEN, 2)
        self.auxiliary_heads = nn.Linear(HIDDEN, len(AUXILIARY_HEADS))

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        values = self.embedding_norm(
            self.token_embeddings(ids) + self.position_embeddings(positions)
        )
        for layer in self.layers:
            values = layer(values, mask)
        pooled = values[:, 0]
        return pooled, self.duration_heads(pooled), self.auxiliary_heads(pooled)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    long_weight: float
    contrastive_weight: float
    focal_gamma: float
    epochs: int
    seed: int


def train_student(
    rows: list[dict],
    teacher_embeddings: np.ndarray,
    tokenizer: WordPiece,
    config: TrainConfig,
    device: torch.device,
    batch_size: int = 64,
) -> SemanticStudent:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == "mps":
        torch.mps.manual_seed(config.seed)
    model = SemanticStudent().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    rng = np.random.default_rng(config.seed)
    order = np.arange(len(rows))
    short_positive = max(1, sum(row.get("final_label") == "short" for row in rows))
    short_weight = min(20.0, (len(rows) - short_positive) / short_positive)
    for _ in range(config.epochs):
        rng.shuffle(order)
        model.train()
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            batch = [rows[int(index)] for index in indices]
            ids, mask = tokenizer.batch_rows(batch, device)
            hidden, duration_logits, auxiliary_logits = model(ids, mask)
            targets = torch.tensor(
                [
                    [
                        float(row.get("final_label") == "short"),
                        float(row.get("final_label") == "long"),
                    ]
                    for row in batch
                ],
                dtype=torch.float32,
                device=device,
            )
            probability = torch.sigmoid(duration_logits)
            bce = F.binary_cross_entropy_with_logits(
                duration_logits, targets, reduction="none"
            )
            pt = probability * targets + (1 - probability) * (1 - targets)
            positive_weights = torch.tensor(
                [short_weight, config.long_weight],
                dtype=torch.float32,
                device=device,
            )
            weights = targets * positive_weights + (1 - targets)
            duration_loss = (bce * (1 - pt).pow(config.focal_gamma) * weights).mean()

            teacher = torch.tensor(
                teacher_embeddings[indices],
                dtype=torch.float32,
                device=device,
            )
            student_similarity = F.normalize(hidden, dim=1) @ F.normalize(hidden, dim=1).T
            teacher_similarity = F.normalize(teacher, dim=1) @ F.normalize(teacher, dim=1).T
            contrastive_loss = F.mse_loss(student_similarity, teacher_similarity)
            auxiliary_loss = _auxiliary_loss(auxiliary_logits, batch, device)
            loss = duration_loss + config.contrastive_weight * contrastive_loss
            loss = loss + 0.2 * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def _auxiliary_loss(
    logits: torch.Tensor, rows: list[dict], device: torch.device
) -> torch.Tensor:
    total = logits.sum() * 0
    terms = 0
    beat_indices = []
    beat_targets = []
    flag_indices: list[tuple[int, int]] = []
    flag_targets = []
    for index, row in enumerate(rows):
        consensus = row.get("structured_consensus") or {}
        beats = consensus.get("beats")
        if beats in {1, 2, 3}:
            beat_indices.append(index)
            beat_targets.append(beats - 1)
        for flag_index, name in enumerate(
            ("ordered", "transformation", "dependent_actions")
        ):
            value = consensus.get(name)
            if type(value) is bool:
                flag_indices.append((index, flag_index + 3))
                flag_targets.append(float(value))
    if beat_indices:
        total = total + F.cross_entropy(
            logits[beat_indices, :3],
            torch.tensor(beat_targets, dtype=torch.long, device=device),
        )
        terms += 1
    if flag_indices:
        selected = torch.stack([logits[row, column] for row, column in flag_indices])
        total = total + F.binary_cross_entropy_with_logits(
            selected,
            torch.tensor(flag_targets, dtype=torch.float32, device=device),
        )
        terms += 1
    return total / max(terms, 1)


def predict(
    model: SemanticStudent,
    rows: list[dict],
    tokenizer: WordPiece,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            ids, mask = tokenizer.batch_rows(batch, device)
            _, logits, _ = model(ids, mask)
            output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def fit_temperatures(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    temperatures = []
    for head in range(2):
        best = (float("inf"), 1.0)
        for temperature in np.linspace(0.25, 4.0, 751):
            values = logits[:, head] / temperature
            loss = np.maximum(values, 0) - values * labels[:, head]
            loss += np.log1p(np.exp(-np.abs(values)))
            candidate = (float(loss.mean()), float(temperature))
            if candidate < best:
                best = candidate
        temperatures.append(best[1])
    return np.asarray(temperatures, dtype=np.float32)


def probabilities(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    values = logits / temperatures[None, :]
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float32)
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def binary_labels(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row.get("final_label") == "short" and not row["unresolved"]),
                float(row.get("final_label") == "long" and not row["unresolved"]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )


def conformal_quantiles(
    values: np.ndarray, labels: np.ndarray, alpha: float = 0.1
) -> np.ndarray:
    output = []
    for head in range(2):
        for positive in (True, False):
            selected = (
                1 - values[labels[:, head] == 1, head]
                if positive
                else values[labels[:, head] == 0, head]
            )
            if not len(selected):
                raise RuntimeError("conformal class is empty")
            rank = min(
                len(selected),
                math.ceil((len(selected) + 1) * (1 - alpha)),
            )
            output.append(float(np.sort(selected)[rank - 1]))
    return np.asarray(output, dtype=np.float32)


def accepted(
    values: np.ndarray, quantiles: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(values, dtype=bool)
    for head in range(2):
        positive = 1 - values[:, head] <= quantiles[head * 2]
        negative = values[:, head] <= quantiles[head * 2 + 1]
        result[:, head] = positive & ~negative & (values[:, head] >= thresholds[head])
    result[result[:, 0] & result[:, 1]] = False
    return result


def select_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    thresholds = []
    targets = (0.95, 0.98)
    for head, target in enumerate(targets):
        best = (0, 1.0)
        for threshold in np.arange(0.8, 0.9951, 0.005):
            current = accepted(
                values,
                quantiles,
                np.asarray(
                    [threshold if head == 0 else 1.0, threshold if head == 1 else 1.0],
                    dtype=np.float32,
                ),
            )[:, head]
            count = int(current.sum())
            precision = (
                float(labels[current, head].mean()) if count else 0.0
            )
            candidate = (count if precision >= target else -1, float(threshold))
            if candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate
        thresholds.append(best[1])
    return np.asarray(thresholds, dtype=np.float32)


def quantize(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = weight.detach().float().cpu().numpy()
    maximum = np.max(np.abs(values), axis=1)
    scales = maximum / 127.0
    safe = np.where(scales == 0, 1.0, scales)
    quantized = np.rint(values / safe[:, None]).clip(-127, 127).astype(np.int8)
    scales = np.where(maximum == 0, 0, scales).astype("<f4")
    return quantized, scales


def quantized_copy(model: SemanticStudent) -> SemanticStudent:
    output = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        for parameter in output.parameters():
            if parameter.ndim != 2:
                continue
            quantized, scales = quantize(parameter)
            dequantized = (
                quantized.astype(np.float32) * scales[:, None]
            )
            parameter.copy_(torch.from_numpy(dequantized))
    return output


def export_v5(
    path: Path,
    model: SemanticStudent,
    vocabulary: list[str],
    metadata: dict,
    temperatures: np.ndarray,
    quantiles: np.ndarray,
    thresholds: np.ndarray,
) -> bytes:
    metadata = {
        **metadata,
        "artifact_format": FORMAT,
        "model_id": "settings-v2",
        "task": "video-duration-selection",
        "labels": ["short", "medium", "long"],
        "policy_compatibility": "quota-safe-v2",
        "min_ngram": 1,
        "max_ngram": 1,
        "bucket": 1,
        "dimension": HIDDEN,
        "model_family": "distilled-semantic-transformer-v1",
        "feature_schema": "wordpiece-transformer-temporal-aux-v1",
        "heads": ["short", "long"],
        "word_min_ngram": 0,
        "word_max_ngram": 0,
        "temporal_features": 0,
        "tokenizer": "wordpiece-v1",
        "vocabulary": vocabulary,
        "max_tokens": MAX_TOKENS,
        "layers": LAYERS,
        "attention_heads": ATTENTION_HEADS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "quantization": "int8-per-output-channel-f32-layernorm",
        "auxiliary_heads": AUXILIARY_HEADS,
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    chunks: list[bytes] = []

    def matrix(weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        quantized, scales = quantize(weight)
        chunks.append(quantized.tobytes(order="C"))
        chunks.append(scales.tobytes(order="C"))
        if bias is not None:
            chunks.append(
                bias.detach().float().cpu().numpy().astype("<f4").tobytes()
            )

    def norm(layer: nn.LayerNorm) -> None:
        chunks.append(layer.weight.detach().float().cpu().numpy().astype("<f4").tobytes())
        chunks.append(layer.bias.detach().float().cpu().numpy().astype("<f4").tobytes())

    matrix(model.token_embeddings.weight)
    matrix(model.position_embeddings.weight)
    norm(model.embedding_norm)
    for layer in model.layers:
        matrix(layer.query.weight, layer.query.bias)
        matrix(layer.key.weight, layer.key.bias)
        matrix(layer.value.weight, layer.value.bias)
        matrix(layer.attention_output.weight, layer.attention_output.bias)
        norm(layer.attention_norm)
        matrix(layer.feed_forward_in.weight, layer.feed_forward_in.bias)
        matrix(layer.feed_forward_out.weight, layer.feed_forward_out.bias)
        norm(layer.output_norm)
    matrix(model.duration_heads.weight, model.duration_heads.bias)
    matrix(model.auxiliary_heads.weight, model.auxiliary_heads.bias)
    chunks.extend(
        values.astype("<f4").tobytes()
        for values in (temperatures, quantiles, thresholds)
    )
    payload = b"".join(chunks)
    header = struct.pack(
        "<IIIIQQ",
        MAGIC,
        FORMAT,
        len(metadata_bytes),
        0,
        len(payload),
        0,
    )
    artifact = header + metadata_bytes + payload
    if len(artifact) >= 32 << 20:
        raise RuntimeError("v5 artifact exceeds 32 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(artifact)
    return artifact
