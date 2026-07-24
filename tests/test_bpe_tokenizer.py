import copy

import numpy as np
import pytest

from localml_scholar.tokenizer import (
    BPETrainingConfig,
    BytePairTokenizer,
    ByteTokenizer,
    MergeRule,
    count_adjacent_pairs,
    load_tokenizer,
    replace_pair_non_overlapping,
)


def test_pair_counting_and_tie_breaking_are_deterministic() -> None:
    assert count_adjacent_pairs([[2, 1, 2], [1, 3]]) == {
        (1, 2): 1,
        (1, 3): 1,
        (2, 1): 1,
    }
    config = BPETrainingConfig(
        target_vocabulary_size=257,
        minimum_pair_frequency=1,
    )

    tokenizer, trace = BytePairTokenizer.train_with_trace("baac", config)

    # (97, 97), (98, 97), and (97, 99) tie at one; lexicographic wins.
    assert tokenizer.merge_rules[0] == MergeRule(256, 97, 97, 0)
    assert trace[0]["selected_pair"] == [97, 97]


def test_pair_replacement_is_left_to_right_and_non_overlapping() -> None:
    assert replace_pair_non_overlapping([97, 97, 97], (97, 97), 256) == [
        256,
        97,
    ]
    assert replace_pair_non_overlapping(
        [1, 2, 1, 2, 1],
        (1, 2),
        9,
    ) == [9, 9, 1]


def test_banana_training_has_hand_checked_merges_and_encoding() -> None:
    config = BPETrainingConfig(
        target_vocabulary_size=259,
        minimum_pair_frequency=2,
    )

    tokenizer = BytePairTokenizer.train("banana bandana", config)

    assert tokenizer.merge_rules == (
        MergeRule(256, 97, 110, 0),
        MergeRule(257, 98, 256, 1),
        MergeRule(258, 256, 97, 2),
    )
    assert np.array_equal(
        tokenizer.encode("banana bandana"),
        np.array([257, 258, 32, 257, 100, 258], dtype=np.int64),
    )
    assert tokenizer.decode(tokenizer.encode("banana bandana")) == "banana bandana"


def test_merge_rank_priority_is_applied_to_current_sequence() -> None:
    tokenizer = BytePairTokenizer(
        [
            MergeRule(256, 97, 98, 0),
            MergeRule(257, 256, 99, 1),
            MergeRule(258, 257, 256, 2),
        ],
        training_config=BPETrainingConfig(
            target_vocabulary_size=259,
            minimum_pair_frequency=1,
        ),
    )

    assert np.array_equal(
        tokenizer.encode("abcab"),
        np.array([258], dtype=np.int64),
    )
    assert tokenizer.decode([258]) == "abcab"


@pytest.mark.parametrize(
    "text",
    ["", "unseen text", "∫ x² dx", "🧠 banana", "中文\n\tcafé"],
)
def test_bpe_round_trip_for_empty_unseen_and_unicode_text(text: str) -> None:
    tokenizer = BytePairTokenizer.train(
        "banana bandana math ∑ emoji 🧠",
        BPETrainingConfig(target_vocabulary_size=270, minimum_pair_frequency=2),
    )
    before = tokenizer.state_dict()

    encoded = tokenizer.encode(text)

    assert encoded.dtype == np.int64
    assert np.all((encoded >= 0) & (encoded < tokenizer.vocabulary_size))
    assert tokenizer.decode(encoded) == text
    assert tokenizer.state_dict() == before


def test_no_merge_bpe_is_byte_tokenization() -> None:
    tokenizer = BytePairTokenizer(
        training_config=BPETrainingConfig(target_vocabulary_size=256)
    )
    text = "bytes 🧠"

    assert np.array_equal(tokenizer.encode(text), ByteTokenizer().encode(text))


def test_document_boundaries_and_validation_text_do_not_affect_merges() -> None:
    config = BPETrainingConfig(
        target_vocabulary_size=257,
        minimum_pair_frequency=1,
    )
    separate = BytePairTokenizer.train(["a", "b"], config)
    joined = BytePairTokenizer.train("ab", config)
    training_only = BytePairTokenizer.train("aaaa", config)
    same_training = BytePairTokenizer.train("aaaa", config)

    assert separate.merge_rules == ()
    assert joined.merge_rules == (MergeRule(256, 97, 98, 0),)
    assert training_only.merge_rules == same_training.merge_rules
    assert training_only.encode("validation-only Z").size > 0


def test_bpe_trains_from_encoded_byte_documents_without_crossing_boundaries() -> None:
    config = BPETrainingConfig(
        target_vocabulary_size=257,
        minimum_pair_frequency=1,
    )
    as_bytes = BytePairTokenizer.train([b"a", b"b"], config)
    as_integer_documents = BytePairTokenizer.train([[97], [98]], config)
    one_encoded_document = BytePairTokenizer.train([97, 98], config)

    assert as_bytes.merge_rules == ()
    assert as_integer_documents.merge_rules == ()
    assert one_encoded_document.merge_rules == (MergeRule(256, 97, 98, 0),)


@pytest.mark.parametrize(
    "corpus",
    [
        [256],
        [-1],
        [1.5, 2.5],
        [[1, 2], ["mixed"]],
        np.ones((1, 1, 1), dtype=np.int64),
    ],
)
def test_bpe_rejects_malformed_encoded_byte_corpora(corpus) -> None:
    with pytest.raises((TypeError, ValueError)):
        BytePairTokenizer.train(
            corpus,
            BPETrainingConfig(target_vocabulary_size=256),
        )


def test_bpe_stopping_conditions() -> None:
    by_frequency = BytePairTokenizer.train(
        "abc",
        BPETrainingConfig(
            target_vocabulary_size=260,
            minimum_pair_frequency=2,
        ),
    )
    by_maximum = BytePairTokenizer.train(
        "aaaaaa",
        BPETrainingConfig(
            target_vocabulary_size=260,
            minimum_pair_frequency=1,
            maximum_merges=1,
        ),
    )
    by_target = BytePairTokenizer.train(
        "aaaaaa",
        BPETrainingConfig(
            target_vocabulary_size=257,
            minimum_pair_frequency=1,
        ),
    )

    assert by_frequency.merge_rules == ()
    assert len(by_maximum.merge_rules) == 1
    assert len(by_target.merge_rules) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_vocabulary_size": 255},
        {"target_vocabulary_size": 256.0},
        {"minimum_pair_frequency": 0},
        {"maximum_merges": 0},
        {"target_vocabulary_size": 257, "maximum_merges": 2},
        {"normalization": "nfc"},
        {"corpus_boundary_policy": "concatenate"},
    ],
)
def test_bpe_configuration_rejects_malformed_values(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        BPETrainingConfig(**kwargs)


def test_bpe_rejects_empty_corpus() -> None:
    config = BPETrainingConfig(target_vocabulary_size=256)
    with pytest.raises(ValueError, match="document"):
        BytePairTokenizer.train([], config)
    with pytest.raises(ValueError, match="UTF-8 byte"):
        BytePairTokenizer.train("", config)


@pytest.mark.parametrize(
    "rules",
    [
        [MergeRule(257, 97, 98, 0)],
        [MergeRule(256, 256, 97, 0)],
        [MergeRule(256, 97, 98, 1)],
        [
            MergeRule(256, 97, 98, 0),
            MergeRule(257, 97, 98, 1),
        ],
    ],
)
def test_bpe_rejects_malformed_merge_tables(rules: list[MergeRule]) -> None:
    with pytest.raises(ValueError):
        BytePairTokenizer(
            rules,
            training_config=BPETrainingConfig(
                target_vocabulary_size=258,
                minimum_pair_frequency=1,
            ),
        )


def test_bpe_serialization_and_transactional_restore(tmp_path) -> None:
    tokenizer = BytePairTokenizer.train(
        "banana bandana",
        BPETrainingConfig(target_vocabulary_size=260, minimum_pair_frequency=1),
    )
    path = tokenizer.save(tmp_path / "bpe.json")
    restored = load_tokenizer(path)

    assert isinstance(restored, BytePairTokenizer)
    assert restored.state_dict() == tokenizer.state_dict()
    assert np.array_equal(
        restored.encode("unseen banana 🧠"),
        tokenizer.encode("unseen banana 🧠"),
    )

    before = tokenizer.state_dict()
    malformed = copy.deepcopy(before)
    malformed["state"]["merge_rules"][0]["left_id"] = 256
    with pytest.raises(ValueError, match="child"):
        tokenizer.load_state_dict(malformed)
    assert tokenizer.state_dict() == before


@pytest.mark.parametrize(
    "metadata",
    [
        {1: "non-string key"},
        {"tuple": (1, 2)},
        {"not_finite": float("nan")},
    ],
)
def test_bpe_rejects_metadata_that_cannot_round_trip_exactly(metadata: dict) -> None:
    with pytest.raises(ValueError, match="JSON"):
        BytePairTokenizer(training_metadata=metadata)


def test_bpe_invalid_ids_and_invalid_utf8_are_explicit() -> None:
    tokenizer = BytePairTokenizer()

    with pytest.raises(ValueError, match="outside"):
        tokenizer.decode([256])
    with pytest.raises(ValueError, match="valid UTF-8"):
        tokenizer.decode([0xFF])
    assert tokenizer.decode([0xFF], errors="replace") == "\ufffd"
