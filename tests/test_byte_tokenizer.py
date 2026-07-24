import copy
import json

import numpy as np
import pytest

from localml_scholar.tokenizer import (
    ByteTokenizer,
    CharacterTokenizer,
    load_tokenizer,
    save_tokenizer,
    tokenizer_from_state_dict,
)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ASCII",
        "café",
        "∑ α→β",
        "emoji: 🧠🚀",
        "中文 العربية हिन्दी",
        "line one\n\tline  two  ",
        "null:\x00:end",
    ],
)
def test_byte_tokenizer_round_trips_arbitrary_unicode(text: str) -> None:
    tokenizer = ByteTokenizer()

    encoded = tokenizer.encode(text)

    assert encoded.dtype == np.int64
    assert np.array_equal(encoded, tokenizer.encode(text))
    assert tokenizer.decode(encoded) == text
    assert np.all((encoded >= 0) & (encoded < 256))


def test_byte_helpers_cover_every_byte_value_exactly() -> None:
    tokenizer = ByteTokenizer()
    raw = bytes(range(256))

    encoded = tokenizer.encode_bytes(raw)

    assert tokenizer.vocabulary_size == 256
    assert np.array_equal(encoded, np.arange(256, dtype=np.int64))
    assert tokenizer.decode_to_bytes(encoded) == raw
    assert all(tokenizer.token_bytes(index) == bytes([index]) for index in range(256))


def test_byte_tokenizer_has_explicit_invalid_utf8_policy() -> None:
    tokenizer = ByteTokenizer()
    invalid = np.array([0xF0, 0x28, 0x8C, 0x28], dtype=np.int64)

    with pytest.raises(ValueError, match="valid UTF-8"):
        tokenizer.decode(invalid)
    assert "\ufffd" in tokenizer.decode(invalid, errors="replace")
    with pytest.raises(ValueError, match="strict.*replace"):
        tokenizer.decode(invalid, errors="ignore")


@pytest.mark.parametrize(
    "values",
    [
        np.array([-1], dtype=np.int64),
        np.array([256], dtype=np.int64),
        np.array([1.0], dtype=np.float64),
        np.array([[1]], dtype=np.int64),
    ],
)
def test_byte_tokenizer_rejects_invalid_token_arrays(values: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        ByteTokenizer().decode(values)


def test_unified_character_and_byte_serialization_is_deterministic(tmp_path) -> None:
    tokenizers = [CharacterTokenizer.from_text("z\naé"), ByteTokenizer()]
    for index, tokenizer in enumerate(tokenizers):
        first = tmp_path / f"first_{index}.json"
        second = tmp_path / f"second_{index}.json"

        save_tokenizer(tokenizer, first)
        save_tokenizer(tokenizer, second)
        restored = load_tokenizer(first)

        assert first.read_bytes() == second.read_bytes()
        assert restored.state_dict() == tokenizer.state_dict()
        assert restored.state_hash() == tokenizer.state_hash()
        assert not list(tmp_path.glob(f".first_{index}.json.*"))
        assert (
            json.loads(first.read_text(encoding="utf-8"))["tokenizer_format_version"]
            == 2
        )


def test_legacy_character_file_migrates_without_changing_ids(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "type": "character",
                "characters": ["\n", "a", "é"],
            }
        ),
        encoding="utf-8",
    )

    tokenizer = load_tokenizer(path)

    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.characters == ("\n", "a", "é")
    assert np.array_equal(tokenizer.encode("é\na"), np.array([2, 0, 1]))
    assert tokenizer.state_dict()["normalization"] == "none"


def test_state_loading_is_transactional_and_unknown_types_fail() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    before = tokenizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["state"]["characters"] = ["b", "a"]

    with pytest.raises(ValueError, match="unique and sorted"):
        tokenizer.load_state_dict(malformed)
    assert tokenizer.state_dict() == before

    unknown = ByteTokenizer().state_dict()
    unknown["tokenizer_type"] = "mystery"
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        tokenizer_from_state_dict(unknown)


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_tokenizer_format_version_requires_an_exact_integer(version) -> None:
    state = ByteTokenizer().state_dict()
    state["tokenizer_format_version"] = version

    with pytest.raises(TypeError, match="must be an integer"):
        tokenizer_from_state_dict(state)
