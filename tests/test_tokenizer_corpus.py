import hashlib

import numpy as np
import pytest

from localml_scholar.data import (
    load_utf8_text,
    prepare_token_stream_dataset,
)
from localml_scholar.tokenizer import (
    BPETrainingConfig,
    BytePairTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
)


@pytest.mark.parametrize("tokenizer_type", ["character", "byte", "bpe"])
def test_split_before_fit_and_boundary_exclusion(tokenizer_type: str) -> None:
    text = "abababababab"
    dataset = prepare_token_stream_dataset(
        text,
        0.5,
        tokenizer=tokenizer_type,
        bpe_config=(
            BPETrainingConfig(
                target_vocabulary_size=258,
                minimum_pair_frequency=2,
            )
            if tokenizer_type == "bpe"
            else None
        ),
    )

    assert dataset.split_index == 6
    assert dataset.tokenizer.decode(dataset.train_tokens) == text[:6]
    assert dataset.tokenizer.decode(dataset.validation_tokens) == text[6:]
    assert dataset.metadata.split_policy.endswith("split_before_tokenizer_fit")
    assert dataset.metadata.train_token_count == dataset.train_tokens.size
    assert dataset.metadata.validation_token_count == dataset.validation_tokens.size


def test_character_vocabulary_and_bpe_merges_use_training_text_only() -> None:
    text = "aaaaaaaaZZZZ"
    with pytest.raises(ValueError, match="absent"):
        prepare_token_stream_dataset(text, 2 / 3, tokenizer="character")
    character = CharacterTokenizer.from_text("aaaaaaaa")
    bpe = prepare_token_stream_dataset(
        text,
        2 / 3,
        tokenizer="bpe",
        bpe_config=BPETrainingConfig(
            target_vocabulary_size=258,
            minimum_pair_frequency=2,
        ),
    )
    training_only_bpe = BytePairTokenizer.train(
        "aaaaaaaa",
        BPETrainingConfig(
            target_vocabulary_size=258,
            minimum_pair_frequency=2,
        ),
    )

    assert character.characters == ("a",)
    assert isinstance(bpe.tokenizer, BytePairTokenizer)
    assert bpe.tokenizer.merge_rules == training_only_bpe.merge_rules
    assert bpe.tokenizer.decode(bpe.validation_tokens) == "ZZZZ"


def test_validation_only_character_fails_but_byte_encodings_succeed() -> None:
    text = "aaaaaa🧠🧠"

    with pytest.raises(ValueError, match="absent"):
        prepare_token_stream_dataset(text, 0.75, tokenizer="character")
    byte = prepare_token_stream_dataset(text, 0.75, tokenizer="byte")

    assert isinstance(byte.tokenizer, ByteTokenizer)
    assert byte.tokenizer.decode(byte.validation_tokens) == "🧠🧠"


def test_prefitted_tokenizer_is_used_without_refitting() -> None:
    tokenizer = BytePairTokenizer.train(
        "broad tokenizer corpus",
        BPETrainingConfig(
            target_vocabulary_size=260,
            minimum_pair_frequency=1,
        ),
    )
    state_before = tokenizer.state_dict()

    dataset = prepare_token_stream_dataset(
        "small train small validation",
        0.6,
        tokenizer=tokenizer,
    )

    assert dataset.tokenizer is tokenizer
    assert tokenizer.state_dict() == state_before


def test_corpus_metadata_hashes_exact_raw_utf8() -> None:
    text = "café\n🧠  "
    dataset = prepare_token_stream_dataset(text, 0.6, tokenizer="byte")
    metadata = dataset.metadata

    assert metadata.character_count == len(text)
    assert metadata.byte_count == len(text.encode("utf-8"))
    assert metadata.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert metadata.tokenizer_state_sha256 == dataset.tokenizer.state_hash()
    assert metadata.to_dict()["normalization"] == "none"


def test_utf8_loader_rejects_empty_and_malformed_files(tmp_path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text("line\n\té", encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    malformed = tmp_path / "malformed.txt"
    malformed.write_bytes(b"\xff\xfe")

    assert load_utf8_text(valid) == "line\n\té"
    with pytest.raises(ValueError, match="empty"):
        load_utf8_text(empty)
    with pytest.raises(ValueError, match="valid UTF-8"):
        load_utf8_text(malformed)


@pytest.mark.parametrize(
    "tokenizer",
    ["unknown", 3, np.array([1], dtype=np.int64)],
)
def test_dataset_rejects_invalid_tokenizer_selection(tokenizer) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_token_stream_dataset("abababab", 0.5, tokenizer=tokenizer)
