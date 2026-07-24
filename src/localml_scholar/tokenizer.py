"""Versioned character, byte, and byte-pair tokenizers."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from localml_scholar.serialization import atomic_write_text

IntArray = NDArray[np.int64]
TOKENIZER_FORMAT_VERSION = 2
NORMALIZATION_NONE = "none"
_DECODE_ERROR_POLICIES = {"strict", "replace"}


def _validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, received {type(text).__name__}.")
    return text


def _validate_decode_errors(errors: str) -> str:
    if not isinstance(errors, str):
        raise TypeError("errors must be a string.")
    if errors not in _DECODE_ERROR_POLICIES:
        raise ValueError("errors must be 'strict' or 'replace'.")
    return errors


def _validate_token_ids(
    token_ids: Sequence[int] | NDArray[np.integer],
    vocabulary_size: int,
) -> IntArray:
    array = np.asarray(token_ids)
    if array.ndim != 1:
        raise ValueError(
            f"token_ids must be one-dimensional, received shape {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("token_ids must contain integers.")
    normalized = array.astype(np.int64, copy=False)
    if normalized.size:
        invalid = np.flatnonzero((normalized < 0) | (normalized >= vocabulary_size))
        if invalid.size:
            position = int(invalid[0])
            token_id = int(normalized[position])
            raise ValueError(
                f"Token ID {token_id} at position {position} is outside "
                f"[0, {vocabulary_size})."
            )
    return normalized


def _canonical_json(state: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(state),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Tokenizer state is not JSON-serializable.") from error


class Tokenizer(ABC):
    """Minimal deterministic text/token contract."""

    tokenizer_type: ClassVar[str]
    normalization: ClassVar[str] = NORMALIZATION_NONE

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Return the number of valid token IDs."""

    @abstractmethod
    def encode(self, text: str) -> IntArray:
        """Encode one text string into a one-dimensional int64 array."""

    @abstractmethod
    def decode(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
        *,
        errors: str = "strict",
    ) -> str:
        """Decode token IDs according to an explicit UTF-8 error policy."""

    @abstractmethod
    def token_bytes(self, token_id: int) -> bytes:
        """Return the UTF-8 bytes represented by one token ID."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return complete versioned JSON-compatible state."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Transactionally replace this tokenizer with validated state."""

    def state_hash(self) -> str:
        """Return a stable SHA-256 digest of canonical tokenizer state."""
        return hashlib.sha256(
            _canonical_json(self.state_dict()).encode("utf-8")
        ).hexdigest()

    def save(self, path: str | Path) -> Path:
        """Atomically save deterministic, human-readable tokenizer JSON."""
        payload = json.dumps(
            self.state_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        return atomic_write_text(path, payload + "\n")


class CharacterTokenizer(Tokenizer):
    """Map Unicode code points to sorted reproducible integer token IDs."""

    FORMAT_VERSION = 1
    tokenizer_type = "character"

    def __init__(self, characters: Iterable[str]) -> None:
        self._set_characters(characters)

    def _set_characters(self, characters: Iterable[str]) -> None:
        materialized = list(characters)
        if not materialized:
            raise ValueError(
                "Tokenizer vocabulary must contain at least one character."
            )
        if any(not isinstance(char, str) or len(char) != 1 for char in materialized):
            raise ValueError("Every vocabulary item must be a one-character string.")
        self._characters = tuple(sorted(set(materialized)))
        self._token_to_index = {
            character: index for index, character in enumerate(self._characters)
        }

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        """Fit a sorted vocabulary to the exact code points in non-empty text."""
        values = _validate_text(text)
        if not values:
            raise ValueError("Cannot build a tokenizer from empty text.")
        return cls(values)

    @property
    def vocabulary_size(self) -> int:
        return len(self._characters)

    @property
    def characters(self) -> tuple[str, ...]:
        """Return token characters in token-ID order."""
        return self._characters

    def encode(self, text: str) -> IntArray:
        values = _validate_text(text)
        encoded = np.empty(len(values), dtype=np.int64)
        for position, character in enumerate(values):
            try:
                encoded[position] = self._token_to_index[character]
            except KeyError as error:
                raise ValueError(
                    f"Unknown character {ascii(character)} at text position "
                    f"{position}; it is not present in the tokenizer vocabulary."
                ) from error
        return encoded

    def decode(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
        *,
        errors: str = "strict",
    ) -> str:
        _validate_decode_errors(errors)
        values = _validate_token_ids(token_ids, self.vocabulary_size)
        return "".join(self._characters[int(token_id)] for token_id in values)

    def token_bytes(self, token_id: int) -> bytes:
        values = _validate_token_ids([token_id], self.vocabulary_size)
        return self._characters[int(values[0])].encode("utf-8")

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_format_version": TOKENIZER_FORMAT_VERSION,
            "tokenizer_type": self.tokenizer_type,
            "normalization": self.normalization,
            "vocabulary_size": self.vocabulary_size,
            "state": {"characters": list(self._characters)},
            "metadata": {},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CharacterTokenizer:
        normalized = _validated_unified_state(state, expected_type=cls.tokenizer_type)
        inner = normalized["state"]
        if set(inner) != {"characters"} or not isinstance(
            inner["characters"],
            list,
        ):
            raise ValueError(
                "Character tokenizer state must contain a characters list."
            )
        tokenizer = cls(inner["characters"])
        if list(tokenizer.characters) != inner["characters"]:
            raise ValueError(
                "Tokenizer characters must be unique and sorted by Unicode code point."
            )
        if normalized["vocabulary_size"] != tokenizer.vocabulary_size:
            raise ValueError("Character tokenizer vocabulary_size is inconsistent.")
        return tokenizer

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        candidate = type(self).from_state_dict(state)
        self._characters = candidate._characters
        self._token_to_index = candidate._token_to_index

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        tokenizer = load_tokenizer(path)
        if not isinstance(tokenizer, cls):
            raise ValueError("Tokenizer JSON type must be 'character'.")
        return tokenizer


class ByteTokenizer(Tokenizer):
    """Fixed tokenizer where IDs 0 through 255 are raw UTF-8 bytes."""

    tokenizer_type = "byte"

    @property
    def vocabulary_size(self) -> int:
        return 256

    def encode(self, text: str) -> IntArray:
        values = _validate_text(text)
        return self.encode_bytes(values.encode("utf-8"))

    def encode_bytes(self, values: bytes | bytearray | memoryview) -> IntArray:
        """Encode an exact byte sequence, including every possible byte value."""
        if not isinstance(values, (bytes, bytearray, memoryview)):
            raise TypeError("values must be bytes-like.")
        raw = bytes(values)
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int64)

    def decode_to_bytes(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
    ) -> bytes:
        values = _validate_token_ids(token_ids, self.vocabulary_size)
        return bytes(values.tolist())

    def decode(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
        *,
        errors: str = "strict",
    ) -> str:
        policy = _validate_decode_errors(errors)
        try:
            return self.decode_to_bytes(token_ids).decode("utf-8", errors=policy)
        except UnicodeDecodeError as error:
            raise ValueError(
                "Token IDs do not form valid UTF-8; use errors='replace' "
                "for a user-facing display."
            ) from error

    def token_bytes(self, token_id: int) -> bytes:
        values = _validate_token_ids([token_id], self.vocabulary_size)
        return bytes([int(values[0])])

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_format_version": TOKENIZER_FORMAT_VERSION,
            "tokenizer_type": self.tokenizer_type,
            "normalization": self.normalization,
            "vocabulary_size": self.vocabulary_size,
            "state": {},
            "metadata": {},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ByteTokenizer:
        normalized = _validated_unified_state(state, expected_type=cls.tokenizer_type)
        if normalized["vocabulary_size"] != 256:
            raise ValueError("Byte tokenizer vocabulary_size must be exactly 256.")
        if normalized["state"] != {} or normalized["metadata"] != {}:
            raise ValueError("Byte tokenizer state and metadata must be empty objects.")
        return cls()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        type(self).from_state_dict(state)

    @classmethod
    def load(cls, path: str | Path) -> ByteTokenizer:
        tokenizer = load_tokenizer(path)
        if not isinstance(tokenizer, cls):
            raise ValueError("Tokenizer JSON type must be 'byte'.")
        return tokenizer


@dataclass(frozen=True)
class MergeRule:
    """One ranked byte-pair merge whose children precede its token ID."""

    token_id: int
    left_id: int
    right_id: int
    rank: int

    def to_dict(self) -> dict[str, int]:
        return {
            "token_id": self.token_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class BPETrainingConfig:
    """Deterministic byte-pair training and stopping configuration."""

    target_vocabulary_size: int = 300
    minimum_pair_frequency: int = 2
    maximum_merges: int | None = None
    normalization: str = NORMALIZATION_NONE
    corpus_boundary_policy: str = "preserve_documents"

    def __post_init__(self) -> None:
        for name in ("target_vocabulary_size", "minimum_pair_frequency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.target_vocabulary_size < 256:
            raise ValueError("target_vocabulary_size must be at least 256.")
        if self.minimum_pair_frequency <= 0:
            raise ValueError("minimum_pair_frequency must be positive.")
        if self.maximum_merges is not None:
            if isinstance(self.maximum_merges, bool) or not isinstance(
                self.maximum_merges,
                int,
            ):
                raise TypeError("maximum_merges must be None or an integer.")
            if self.maximum_merges <= 0:
                raise ValueError("maximum_merges must be positive when supplied.")
            available = self.target_vocabulary_size - 256
            if self.maximum_merges > available:
                raise ValueError(
                    "maximum_merges cannot exceed target_vocabulary_size - 256."
                )
        if self.normalization != NORMALIZATION_NONE:
            raise ValueError("Only normalization='none' is supported.")
        if self.corpus_boundary_policy != "preserve_documents":
            raise ValueError(
                "Only corpus_boundary_policy='preserve_documents' is supported."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_vocabulary_size": self.target_vocabulary_size,
            "minimum_pair_frequency": self.minimum_pair_frequency,
            "maximum_merges": self.maximum_merges,
            "normalization": self.normalization,
            "corpus_boundary_policy": self.corpus_boundary_policy,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BPETrainingConfig:
        if not isinstance(values, Mapping):
            raise TypeError("BPE training configuration must be a mapping.")
        expected = set(cls().to_dict())
        if set(values) != expected:
            raise ValueError("BPE training configuration keys are malformed.")
        return cls(**dict(values))


def count_adjacent_pairs(
    sequences: Sequence[Sequence[int]],
) -> dict[tuple[int, int], int]:
    """Count adjacent pairs without crossing sequence/document boundaries."""
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:], strict=False))
    return dict(sorted(counts.items()))


def replace_pair_non_overlapping(
    sequence: Sequence[int],
    pair: tuple[int, int],
    token_id: int,
) -> list[int]:
    """Replace a pair left-to-right without overlapping occurrences."""
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
    ):
        raise TypeError("pair must contain exactly two integer token IDs.")
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError("token_id must be an integer.")
    output: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            output.append(token_id)
            index += 2
        else:
            output.append(int(sequence[index]))
            index += 1
    return output


def _validated_byte_document(document: object) -> list[int]:
    if isinstance(document, (bytes, bytearray, memoryview)):
        return list(bytes(document))
    if isinstance(document, str):
        raise TypeError(
            "Encoded BPE documents must contain byte IDs, not mixed text strings."
        )
    try:
        values = np.asarray(document)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Each encoded BPE document must be a one-dimensional byte sequence."
        ) from error
    if values.ndim != 1:
        raise ValueError(
            "Each encoded BPE document must be one-dimensional, "
            f"received shape {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("Encoded BPE documents must contain integer byte IDs.")
    if values.size and (np.any(values < 0) or np.any(values > 255)):
        raise ValueError("Encoded BPE document IDs must lie in [0, 255].")
    return values.astype(np.uint8, copy=False).tolist()


def _bpe_corpus_to_byte_sequences(corpus: object) -> list[list[int]]:
    if isinstance(corpus, str):
        sequences = [list(corpus.encode("utf-8"))]
    elif isinstance(corpus, (bytes, bytearray, memoryview)):
        sequences = [list(bytes(corpus))]
    elif isinstance(corpus, np.ndarray):
        if corpus.ndim == 1:
            sequences = [_validated_byte_document(corpus)]
        elif corpus.ndim == 2:
            sequences = [_validated_byte_document(document) for document in corpus]
        else:
            raise ValueError(
                "Encoded BPE corpus arrays must have one or two dimensions."
            )
    elif isinstance(corpus, Sequence):
        documents = list(corpus)
        if not documents:
            raise ValueError("BPE corpus must contain at least one document.")
        if all(isinstance(document, str) for document in documents):
            sequences = [list(document.encode("utf-8")) for document in documents]
        elif all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in documents
        ):
            sequences = [_validated_byte_document(documents)]
        else:
            sequences = [_validated_byte_document(document) for document in documents]
    else:
        raise TypeError(
            "corpus must be text, bytes, or a sequence of text/byte documents."
        )
    if sum(len(sequence) for sequence in sequences) == 0:
        raise ValueError("BPE corpus must contain at least one UTF-8 byte.")
    return sequences


class BytePairTokenizer(Tokenizer):
    """Deterministic byte-level BPE with contiguous ranked merge IDs."""

    tokenizer_type = "bpe"

    def __init__(
        self,
        merge_rules: Sequence[MergeRule] = (),
        *,
        training_config: BPETrainingConfig | None = None,
        training_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.training_config = training_config or BPETrainingConfig(
            target_vocabulary_size=256
        )
        self._merge_rules = self._validate_merge_rules(merge_rules)
        if len(self._merge_rules) > (self.training_config.target_vocabulary_size - 256):
            raise ValueError("Merge rules exceed the configured target vocabulary.")
        metadata = {} if training_metadata is None else dict(training_metadata)
        serialized_metadata = _canonical_json(metadata)
        if json.loads(serialized_metadata) != metadata:
            raise ValueError(
                "training_metadata must use exact JSON object, array, scalar, "
                "and string-key types."
            )
        self._training_metadata = metadata
        self._build_tables()

    @staticmethod
    def _validate_merge_rules(
        merge_rules: Sequence[MergeRule],
    ) -> tuple[MergeRule, ...]:
        if isinstance(merge_rules, (str, bytes)) or not isinstance(
            merge_rules,
            Sequence,
        ):
            raise TypeError("merge_rules must be a sequence of MergeRule objects.")
        validated: list[MergeRule] = []
        seen_pairs: set[tuple[int, int]] = set()
        for expected_rank, rule in enumerate(merge_rules):
            if not isinstance(rule, MergeRule):
                raise TypeError("Every merge rule must be a MergeRule.")
            expected_token_id = 256 + expected_rank
            for name in ("token_id", "left_id", "right_id", "rank"):
                value = getattr(rule, name)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"Merge rule {name} must be an integer.")
            if rule.rank != expected_rank:
                raise ValueError("Merge ranks must be contiguous from zero.")
            if rule.token_id != expected_token_id:
                raise ValueError("Merge token IDs must be contiguous from 256.")
            if not 0 <= rule.left_id < rule.token_id:
                raise ValueError("Merge left child must exist before its parent.")
            if not 0 <= rule.right_id < rule.token_id:
                raise ValueError("Merge right child must exist before its parent.")
            pair = (rule.left_id, rule.right_id)
            if pair in seen_pairs:
                raise ValueError("Duplicate merge pairs are not allowed.")
            seen_pairs.add(pair)
            validated.append(rule)
        return tuple(validated)

    def _build_tables(self) -> None:
        self._pair_to_rule = {
            (rule.left_id, rule.right_id): rule for rule in self._merge_rules
        }
        expansions: list[bytes] = [bytes([value]) for value in range(256)]
        for rule in self._merge_rules:
            expansions.append(expansions[rule.left_id] + expansions[rule.right_id])
        self._expansions = tuple(expansions)

    @property
    def merge_rules(self) -> tuple[MergeRule, ...]:
        return self._merge_rules

    @property
    def vocabulary_size(self) -> int:
        return 256 + len(self._merge_rules)

    @classmethod
    def train(
        cls,
        corpus: (
            str
            | bytes
            | bytearray
            | memoryview
            | Sequence[str]
            | Sequence[int]
            | Sequence[Sequence[int] | bytes | bytearray | memoryview]
        ),
        config: BPETrainingConfig,
    ) -> BytePairTokenizer:
        """Fit deterministic merges to text or encoded byte documents."""
        tokenizer, _ = cls.train_with_trace(corpus, config)
        return tokenizer

    @classmethod
    def train_with_trace(
        cls,
        corpus: (
            str
            | bytes
            | bytearray
            | memoryview
            | Sequence[str]
            | Sequence[int]
            | Sequence[Sequence[int] | bytes | bytearray | memoryview]
        ),
        config: BPETrainingConfig,
    ) -> tuple[BytePairTokenizer, list[dict[str, Any]]]:
        """Fit BPE and return transparent per-merge training trace."""
        if not isinstance(config, BPETrainingConfig):
            raise TypeError("config must be a BPETrainingConfig.")
        sequences = _bpe_corpus_to_byte_sequences(corpus)
        total_bytes = sum(len(sequence) for sequence in sequences)

        rules: list[MergeRule] = []
        trace: list[dict[str, Any]] = []
        maximum_by_target = config.target_vocabulary_size - 256
        merge_limit = (
            maximum_by_target
            if config.maximum_merges is None
            else min(maximum_by_target, config.maximum_merges)
        )
        stop_reason = "target_vocabulary_size"
        while len(rules) < merge_limit:
            counts = count_adjacent_pairs(sequences)
            eligible = {
                pair: frequency
                for pair, frequency in counts.items()
                if frequency >= config.minimum_pair_frequency
            }
            if not eligible:
                stop_reason = "no_eligible_pair"
                break
            selected_pair, selected_frequency = min(
                eligible.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            token_id = 256 + len(rules)
            rule = MergeRule(
                token_id=token_id,
                left_id=selected_pair[0],
                right_id=selected_pair[1],
                rank=len(rules),
            )
            sequences = [
                replace_pair_non_overlapping(sequence, selected_pair, token_id)
                for sequence in sequences
            ]
            rules.append(rule)
            trace.append(
                {
                    "rank": rule.rank,
                    "pair_counts": [
                        {"pair": [left, right], "frequency": frequency}
                        for (left, right), frequency in counts.items()
                    ],
                    "selected_pair": list(selected_pair),
                    "selected_frequency": selected_frequency,
                    "new_token_id": token_id,
                    "sequences_after": [list(sequence) for sequence in sequences],
                }
            )
        if config.maximum_merges is not None and len(rules) == merge_limit:
            stop_reason = "maximum_merges"
        metadata = {
            "document_count": len(sequences),
            "training_byte_count": total_bytes,
            "learned_merge_count": len(rules),
            "stop_reason": stop_reason,
        }
        return (
            cls(
                rules,
                training_config=config,
                training_metadata=metadata,
            ),
            trace,
        )

    def encode(self, text: str) -> IntArray:
        values = _validate_text(text)
        sequence = list(values.encode("utf-8"))
        while len(sequence) >= 2:
            available = [
                self._pair_to_rule[pair]
                for pair in zip(sequence, sequence[1:], strict=False)
                if pair in self._pair_to_rule
            ]
            if not available:
                break
            selected = min(available, key=lambda rule: rule.rank)
            sequence = replace_pair_non_overlapping(
                sequence,
                (selected.left_id, selected.right_id),
                selected.token_id,
            )
        return np.asarray(sequence, dtype=np.int64)

    def decode_to_bytes(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
    ) -> bytes:
        values = _validate_token_ids(token_ids, self.vocabulary_size)
        return b"".join(self._expansions[int(token_id)] for token_id in values)

    def decode(
        self,
        token_ids: Sequence[int] | NDArray[np.integer],
        *,
        errors: str = "strict",
    ) -> str:
        policy = _validate_decode_errors(errors)
        try:
            return self.decode_to_bytes(token_ids).decode("utf-8", errors=policy)
        except UnicodeDecodeError as error:
            raise ValueError(
                "Token IDs do not form valid UTF-8; use errors='replace' "
                "for a user-facing display."
            ) from error

    def token_bytes(self, token_id: int) -> bytes:
        values = _validate_token_ids([token_id], self.vocabulary_size)
        return self._expansions[int(values[0])]

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_format_version": TOKENIZER_FORMAT_VERSION,
            "tokenizer_type": self.tokenizer_type,
            "normalization": self.normalization,
            "vocabulary_size": self.vocabulary_size,
            "state": {
                "training_config": self.training_config.to_dict(),
                "merge_rules": [rule.to_dict() for rule in self._merge_rules],
            },
            "metadata": dict(self._training_metadata),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> BytePairTokenizer:
        normalized = _validated_unified_state(state, expected_type=cls.tokenizer_type)
        inner = normalized["state"]
        if set(inner) != {"training_config", "merge_rules"}:
            raise ValueError("BPE tokenizer state keys are malformed.")
        config = BPETrainingConfig.from_dict(inner["training_config"])
        serialized_rules = inner["merge_rules"]
        if not isinstance(serialized_rules, list):
            raise ValueError("BPE merge_rules must be a list.")
        rules: list[MergeRule] = []
        for serialized in serialized_rules:
            if not isinstance(serialized, Mapping) or set(serialized) != {
                "token_id",
                "left_id",
                "right_id",
                "rank",
            }:
                raise ValueError("A serialized BPE merge rule is malformed.")
            rules.append(MergeRule(**dict(serialized)))
        metadata = normalized["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("BPE tokenizer metadata must be an object.")
        tokenizer = cls(
            rules,
            training_config=config,
            training_metadata=metadata,
        )
        if normalized["vocabulary_size"] != tokenizer.vocabulary_size:
            raise ValueError("BPE tokenizer vocabulary_size is inconsistent.")
        return tokenizer

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        candidate = type(self).from_state_dict(state)
        self.training_config = candidate.training_config
        self._merge_rules = candidate._merge_rules
        self._training_metadata = candidate._training_metadata
        self._build_tables()

    @classmethod
    def load(cls, path: str | Path) -> BytePairTokenizer:
        tokenizer = load_tokenizer(path)
        if not isinstance(tokenizer, cls):
            raise ValueError("Tokenizer JSON type must be 'bpe'.")
        return tokenizer


def _validated_unified_state(
    state: Mapping[str, Any],
    *,
    expected_type: str | None = None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("Tokenizer state must be a mapping.")
    expected_keys = {
        "tokenizer_format_version",
        "tokenizer_type",
        "normalization",
        "vocabulary_size",
        "state",
        "metadata",
    }
    if set(state) != expected_keys:
        raise ValueError("Tokenizer state keys do not match the versioned format.")
    normalized = dict(state)
    format_version = normalized["tokenizer_format_version"]
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise TypeError("tokenizer_format_version must be an integer.")
    if format_version != TOKENIZER_FORMAT_VERSION:
        raise ValueError(f"Unsupported tokenizer format version: {format_version!r}.")
    tokenizer_type = normalized["tokenizer_type"]
    if not isinstance(tokenizer_type, str):
        raise TypeError("tokenizer_type must be a string.")
    if expected_type is not None and tokenizer_type != expected_type:
        raise ValueError(
            f"Tokenizer type must be {expected_type!r}, got {tokenizer_type!r}."
        )
    if normalized["normalization"] != NORMALIZATION_NONE:
        raise ValueError("Only tokenizer normalization='none' is supported.")
    vocabulary_size = normalized["vocabulary_size"]
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        raise TypeError("Tokenizer vocabulary_size must be an integer.")
    if vocabulary_size <= 0:
        raise ValueError("Tokenizer vocabulary_size must be positive.")
    if not isinstance(normalized["state"], dict):
        raise ValueError("Tokenizer state field must be an object.")
    if not isinstance(normalized["metadata"], dict):
        raise ValueError("Tokenizer metadata field must be an object.")
    return normalized


def migrate_legacy_character_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Convert legacy v1 character metadata without changing token order."""
    if not isinstance(state, Mapping):
        raise TypeError("Legacy tokenizer state must be a mapping.")
    expected = {"format_version", "type", "characters"}
    if set(state) != expected:
        raise ValueError("Legacy character tokenizer state keys are malformed.")
    format_version = state["format_version"]
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise TypeError("Legacy tokenizer format_version must be an integer.")
    if format_version != CharacterTokenizer.FORMAT_VERSION:
        raise ValueError(
            f"Unsupported legacy tokenizer format version: {format_version!r}."
        )
    if state["type"] != CharacterTokenizer.tokenizer_type:
        raise ValueError("Legacy tokenizer type must be 'character'.")
    characters = state["characters"]
    if not isinstance(characters, list):
        raise ValueError("Legacy tokenizer characters must be a list.")
    tokenizer = CharacterTokenizer(characters)
    if list(tokenizer.characters) != characters:
        raise ValueError(
            "Legacy tokenizer characters must be unique and sorted by code point."
        )
    return tokenizer.state_dict()


def tokenizer_from_state_dict(state: Mapping[str, Any]) -> Tokenizer:
    """Construct the concrete tokenizer from unified or legacy state."""
    if not isinstance(state, Mapping):
        raise TypeError("Tokenizer state must be a mapping.")
    normalized: Mapping[str, Any]
    if set(state) == {"format_version", "type", "characters"}:
        normalized = migrate_legacy_character_state(state)
    else:
        normalized = state
    validated = _validated_unified_state(normalized)
    tokenizer_type = validated["tokenizer_type"]
    if tokenizer_type == CharacterTokenizer.tokenizer_type:
        return CharacterTokenizer.from_state_dict(validated)
    if tokenizer_type == ByteTokenizer.tokenizer_type:
        return ByteTokenizer.from_state_dict(validated)
    if tokenizer_type == BytePairTokenizer.tokenizer_type:
        return BytePairTokenizer.from_state_dict(validated)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type!r}.")


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> Path:
    """Atomically save any supported tokenizer."""
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must implement the Tokenizer interface.")
    return tokenizer.save(path)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load any supported tokenizer from versioned JSON or legacy character JSON."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Tokenizer file does not exist: {source}") from None
    except UnicodeDecodeError as error:
        raise ValueError(f"Tokenizer file is not valid UTF-8: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Tokenizer file is not valid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer JSON must contain an object.")
    return tokenizer_from_state_dict(payload)
