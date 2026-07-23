import json

import numpy as np
import pytest

from localml_scholar.data import MiniBatchSampler, prepare_bigram_dataset
from localml_scholar.tokenizer import CharacterTokenizer


def test_encode_decode_round_trip() -> None:
    text = "cab\ncab"
    tokenizer = CharacterTokenizer.from_text(text)

    encoded = tokenizer.encode(text)

    assert encoded.dtype == np.int64
    assert tokenizer.decode(encoded) == text


def test_vocabulary_is_sorted_and_deterministic() -> None:
    first = CharacterTokenizer.from_text("zbaéa")
    second = CharacterTokenizer.from_text("aézb")

    assert first.characters == tuple(sorted(set("zbaé")))
    assert first.characters == second.characters
    assert np.array_equal(first.encode("éabz"), second.encode("éabz"))


def test_save_and_load_preserve_vocabulary(tmp_path) -> None:
    tokenizer = CharacterTokenizer.from_text("paper\n∑")
    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)

    assert loaded.characters == tokenizer.characters
    assert np.array_equal(loaded.encode("∑paper"), tokenizer.encode("∑paper"))


def test_unknown_character_has_clear_error() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match=r"Unknown character 'z'.*position 1"):
        tokenizer.encode("az")


def test_malformed_serialization_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "type": "character",
                "characters": ["b", "a"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique and sorted"):
        CharacterTokenizer.load(path)


def test_chronological_split_discards_boundary_pair() -> None:
    dataset = prepare_bigram_dataset("ababababab", train_fraction=0.6)

    assert dataset.split_index == 6
    assert dataset.train_inputs.size == 5
    assert dataset.validation_inputs.size == 3
    assert dataset.tokenizer.decode(dataset.train_inputs) == "ababa"
    assert dataset.tokenizer.decode(dataset.validation_inputs) == "aba"


def test_validation_only_character_does_not_leak_into_vocabulary() -> None:
    with pytest.raises(ValueError, match="Validation text contains characters"):
        prepare_bigram_dataset("aaaaaZ", train_fraction=0.75)


def test_minibatches_are_reproducible() -> None:
    inputs = np.arange(8, dtype=np.int64)
    targets = np.roll(inputs, -1)
    first = MiniBatchSampler(inputs, targets, batch_size=5, seed=17)
    second = MiniBatchSampler(inputs, targets, batch_size=5, seed=17)

    for _ in range(3):
        first_batch = first.next_batch()
        second_batch = second.next_batch()
        assert np.array_equal(first_batch[0], second_batch[0])
        assert np.array_equal(first_batch[1], second_batch[1])


def test_invalid_dataset_and_sampler_inputs_raise() -> None:
    with pytest.raises(ValueError, match="four characters"):
        prepare_bigram_dataset("abc", train_fraction=0.5)
    with pytest.raises(ValueError, match="strictly between"):
        prepare_bigram_dataset("abab", train_fraction=1.0)
    with pytest.raises(ValueError, match="identical shapes"):
        MiniBatchSampler(
            np.array([0, 1], dtype=np.int64),
            np.array([1], dtype=np.int64),
            batch_size=1,
            seed=0,
        )
