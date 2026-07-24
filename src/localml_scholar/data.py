"""Local text loading, chronological splitting, and reproducible minibatches."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from localml_scholar.tokenizer import CharacterTokenizer

IntArray = NDArray[np.int64]

FALLBACK_CORPUS = (
    "local models learn from local text.\n"
    "a bigram predicts one character from the previous character.\n"
) * 24


@dataclass(frozen=True)
class BigramDataset:
    """Chronological train/validation bigram examples and their tokenizer."""

    tokenizer: CharacterTokenizer
    train_inputs: IntArray
    train_targets: IntArray
    validation_inputs: IntArray
    validation_targets: IntArray
    split_index: int


@dataclass(frozen=True)
class TokenStreamDataset:
    """Chronologically isolated token streams for sequence language modeling."""

    tokenizer: CharacterTokenizer
    train_tokens: IntArray
    validation_tokens: IntArray
    split_index: int


def load_utf8_text(path: str | Path) -> str:
    """Load a non-empty local UTF-8 text file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input corpus does not exist: {source}") from None
    except UnicodeDecodeError as error:
        raise ValueError(f"Input corpus is not valid UTF-8: {source}") from error
    if not text:
        raise ValueError(f"Input corpus is empty: {source}")
    return text


def validate_train_fraction(train_fraction: float) -> None:
    """Validate a chronological train fraction."""
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)):
        raise TypeError("train_fraction must be a real number.")
    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and strictly between 0 and 1.")


def next_token_pairs(token_ids: IntArray) -> tuple[IntArray, IntArray]:
    """Return ``(x_t, x_{t+1})`` examples from a token sequence."""
    array = np.asarray(token_ids)
    if array.ndim != 1:
        raise ValueError(
            f"token_ids must be one-dimensional, received shape {array.shape}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("token_ids must contain integers.")
    if array.size < 2:
        raise ValueError("At least two tokens are required to form a bigram pair.")
    normalized = array.astype(np.int64, copy=False)
    return normalized[:-1].copy(), normalized[1:].copy()


def prepare_bigram_dataset(text: str, train_fraction: float) -> BigramDataset:
    """Split text chronologically, then create non-overlapping split examples.

    The tokenizer is fit only on the training text. Validation characters not
    observed during training therefore produce an explicit error instead of
    leaking validation vocabulary information into training.
    """
    dataset = prepare_token_stream_dataset(text, train_fraction)
    train_inputs, train_targets = next_token_pairs(dataset.train_tokens)
    validation_inputs, validation_targets = next_token_pairs(dataset.validation_tokens)
    return BigramDataset(
        tokenizer=dataset.tokenizer,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        split_index=dataset.split_index,
    )


def prepare_token_stream_dataset(
    text: str,
    train_fraction: float,
) -> TokenStreamDataset:
    """Create isolated chronological token streams with a train-only vocabulary.

    The character at ``split_index - 1`` belongs only to training and the
    character at ``split_index`` belongs only to validation. Sequence samplers
    operate on these streams independently, so no boundary target is created.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, received {type(text).__name__}.")
    validate_train_fraction(train_fraction)
    if len(text) < 4:
        raise ValueError(
            "At least four characters are required so both splits contain "
            "a next-token example."
        )

    split_index = int(len(text) * float(train_fraction))
    if split_index < 2 or len(text) - split_index < 2:
        raise ValueError(
            "train_fraction leaves fewer than two characters in a split; "
            "choose a longer corpus or a less extreme fraction."
        )

    train_text = text[:split_index]
    validation_text = text[split_index:]
    tokenizer = CharacterTokenizer.from_text(train_text)
    train_tokens = tokenizer.encode(train_text)
    try:
        validation_tokens = tokenizer.encode(validation_text)
    except ValueError as error:
        raise ValueError(
            "Validation text contains characters absent from the chronological "
            "training split. Choose a representative training split or handle "
            "unknown tokens in a later tokenizer milestone."
        ) from error
    return TokenStreamDataset(
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        split_index=split_index,
    )


class MiniBatchSampler:
    """Draw reproducible random minibatches with replacement."""

    def __init__(
        self,
        inputs: IntArray,
        targets: IntArray,
        batch_size: int,
        seed: int,
    ) -> None:
        self.inputs = self._validate_array(inputs, "inputs")
        self.targets = self._validate_array(targets, "targets")
        if self.inputs.shape != self.targets.shape:
            raise ValueError(
                "inputs and targets must have identical shapes, received "
                f"{self.inputs.shape} and {self.targets.shape}."
            )
        if self.inputs.size == 0:
            raise ValueError("Cannot sample from an empty dataset.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _validate_array(values: IntArray, name: str) -> IntArray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must contain integers.")
        return array.astype(np.int64, copy=False)

    @property
    def num_examples(self) -> int:
        """Return the number of available examples."""
        return int(self.inputs.size)

    def next_batch(self) -> tuple[IntArray, IntArray]:
        """Sample one batch using this sampler's private RNG."""
        indices = self._rng.integers(
            0, self.inputs.size, size=self.batch_size, endpoint=False
        )
        return self.inputs[indices], self.targets[indices]


class SequenceBatchSampler:
    """Sample shifted fixed-length sequences with a private restorable RNG.

    Sampling is uniform over all valid start positions and uses replacement.
    For a stream of length ``N`` and sequence length ``T``, starts are drawn
    from ``[0, N - T)`` so every target at ``start + T`` remains in the stream.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        token_ids: NDArray[np.integer],
        *,
        batch_size: int,
        sequence_length: int,
        seed: int,
    ) -> None:
        values = np.asarray(token_ids)
        if values.ndim != 1:
            raise ValueError(
                f"token_ids must be one-dimensional, got shape {values.shape}."
            )
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("token_ids must contain integers.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if values.size <= sequence_length:
            raise ValueError(
                f"Token stream length {values.size} must exceed sequence length "
                f"{sequence_length}."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")

        self.token_ids = np.array(values, dtype=np.int64, copy=True)
        self.token_ids.setflags(write=False)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._stream_digest = hashlib.sha256(
            self.token_ids.tobytes(order="C")
        ).hexdigest()

    @property
    def valid_start_count(self) -> int:
        """Return the number of starts whose final target stays in the stream."""
        return int(self.token_ids.size - self.sequence_length)

    def batch_from_starts(
        self,
        starts: NDArray[np.integer],
    ) -> tuple[IntArray, IntArray]:
        """Construct shifted inputs and targets from explicitly chosen starts."""
        values = np.asarray(starts)
        if values.ndim != 1:
            raise ValueError(f"starts must be one-dimensional, got {values.shape}.")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("starts must contain integers.")
        if values.size == 0:
            raise ValueError("starts must be non-empty.")
        if np.any(values < 0) or np.any(values >= self.valid_start_count):
            raise ValueError(f"starts must lie in [0, {self.valid_start_count}).")
        offsets = np.arange(self.sequence_length, dtype=np.int64)
        indices = values.astype(np.int64, copy=False)[:, None] + offsets[None, :]
        return (
            np.array(self.token_ids[indices], copy=True),
            np.array(self.token_ids[indices + 1], copy=True),
        )

    def next_batch(self) -> tuple[IntArray, IntArray]:
        """Sample one batch uniformly with replacement."""
        starts = self._rng.integers(
            0,
            self.valid_start_count,
            size=self.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        return self.batch_from_starts(starts)

    def state_dict(self) -> dict[str, Any]:
        """Return complete JSON-serializable sampler state."""
        return {
            "state_version": self.STATE_VERSION,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "stream_length": int(self.token_ids.size),
            "stream_sha256": self._stream_digest,
            "seed": self.seed,
            "bit_generator": type(self._rng.bit_generator).__name__,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a compatible sampler state without partial mutation."""
        if not isinstance(state, Mapping):
            raise TypeError("Sampler state must be a mapping.")
        expected_keys = {
            "state_version",
            "batch_size",
            "sequence_length",
            "stream_length",
            "stream_sha256",
            "seed",
            "bit_generator",
            "rng_state",
        }
        if set(state) != expected_keys:
            raise ValueError("Sampler state keys do not match the expected format.")
        expected_values = {
            "state_version": self.STATE_VERSION,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "stream_length": int(self.token_ids.size),
            "stream_sha256": self._stream_digest,
            "seed": self.seed,
            "bit_generator": type(self._rng.bit_generator).__name__,
        }
        for name, expected in expected_values.items():
            if state[name] != expected:
                raise ValueError(
                    f"Sampler state {name!r} is incompatible: "
                    f"expected {expected!r}, got {state[name]!r}."
                )
        candidate = np.random.default_rng()
        try:
            candidate.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("Sampler RNG state is malformed.") from error
        self._rng.bit_generator.state = copy.deepcopy(candidate.bit_generator.state)
