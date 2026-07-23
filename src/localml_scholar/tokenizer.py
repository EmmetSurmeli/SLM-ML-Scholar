"""Deterministic character-level tokenization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


class CharacterTokenizer:
    """Map Unicode characters to reproducible integer token IDs.

    Characters are always sorted by Unicode code point. This makes the
    vocabulary independent of the order in which characters first appear.
    """

    FORMAT_VERSION = 1

    def __init__(self, characters: Iterable[str]) -> None:
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
        """Build a tokenizer from all unique characters in ``text``."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, received {type(text).__name__}.")
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text.")
        return cls(text)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of distinct characters."""
        return len(self._characters)

    @property
    def characters(self) -> tuple[str, ...]:
        """Return token characters in token-ID order."""
        return self._characters

    def encode(self, text: str) -> IntArray:
        """Encode text as a one-dimensional int64 array.

        Raises:
            ValueError: If ``text`` contains a character outside the vocabulary.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, received {type(text).__name__}.")

        encoded = np.empty(len(text), dtype=np.int64)
        for position, character in enumerate(text):
            try:
                encoded[position] = self._token_to_index[character]
            except KeyError as error:
                printable = ascii(character)
                raise ValueError(
                    f"Unknown character {printable} at text position {position}; "
                    "it is not present in the tokenizer vocabulary."
                ) from error
        return encoded

    def decode(self, token_ids: Sequence[int] | NDArray[np.integer]) -> str:
        """Decode a one-dimensional sequence of token IDs into text."""
        array = np.asarray(token_ids)
        if array.ndim != 1:
            raise ValueError(
                f"token_ids must be one-dimensional, received shape {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("token_ids must contain integers.")

        characters: list[str] = []
        for position, token_id_value in enumerate(array):
            token_id = int(token_id_value)
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError(
                    f"Token ID {token_id} at position {position} is outside "
                    f"[0, {self.vocabulary_size})."
                )
            characters.append(self._characters[token_id])
        return "".join(characters)

    def save(self, path: str | Path) -> Path:
        """Save the vocabulary to a UTF-8 JSON file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "type": "character",
            "characters": list(self._characters),
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        """Load and validate a vocabulary JSON file."""
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tokenizer file does not exist: {source}"
            ) from None
        except json.JSONDecodeError as error:
            raise ValueError(f"Tokenizer file is not valid JSON: {source}") from error

        if not isinstance(payload, dict):
            raise ValueError("Tokenizer JSON must contain an object.")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported tokenizer format version: "
                f"{payload.get('format_version')!r}."
            )
        if payload.get("type") != "character":
            raise ValueError("Tokenizer JSON type must be 'character'.")
        characters = payload.get("characters")
        if not isinstance(characters, list):
            raise ValueError("Tokenizer JSON 'characters' must be a list.")

        tokenizer = cls(characters)
        if list(tokenizer.characters) != characters:
            raise ValueError(
                "Tokenizer characters must be unique and sorted by Unicode code point."
            )
        return tokenizer
