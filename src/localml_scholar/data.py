"""Local text loading, chronological splitting, and reproducible minibatches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    if not isinstance(text, str):
        raise TypeError(f"text must be str, received {type(text).__name__}.")
    validate_train_fraction(train_fraction)
    if len(text) < 4:
        raise ValueError(
            "At least four characters are required so both splits contain a bigram."
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

    train_inputs, train_targets = next_token_pairs(train_tokens)
    validation_inputs, validation_targets = next_token_pairs(validation_tokens)
    return BigramDataset(
        tokenizer=tokenizer,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
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
