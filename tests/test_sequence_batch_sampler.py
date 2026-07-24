import copy

import numpy as np
import pytest

from localml_scholar.data import (
    SequenceBatchSampler,
    prepare_token_stream_dataset,
)


def test_sequence_batch_from_starts_has_exact_shift_and_shape() -> None:
    tokens = np.arange(10, dtype=np.int64)
    sampler = SequenceBatchSampler(
        tokens,
        batch_size=2,
        sequence_length=3,
        seed=4,
    )

    inputs, targets = sampler.batch_from_starts(np.array([0, 6], dtype=np.int64))

    assert inputs.dtype == np.int64
    assert targets.dtype == np.int64
    assert inputs.shape == targets.shape == (2, 3)
    assert np.array_equal(inputs, np.array([[0, 1, 2], [6, 7, 8]]))
    assert np.array_equal(targets, np.array([[1, 2, 3], [7, 8, 9]]))


def test_sequence_sampler_is_seeded_and_different_seeds_diverge() -> None:
    tokens = np.arange(30, dtype=np.int64)
    first = SequenceBatchSampler(tokens, batch_size=8, sequence_length=4, seed=7)
    second = SequenceBatchSampler(tokens, batch_size=8, sequence_length=4, seed=7)
    different = SequenceBatchSampler(tokens, batch_size=8, sequence_length=4, seed=8)

    first_batch = first.next_batch()

    assert all(
        np.array_equal(left, right)
        for left, right in zip(first_batch, second.next_batch(), strict=True)
    )
    assert not np.array_equal(first_batch[0], different.next_batch()[0])


def test_sequence_sampler_state_restores_exact_next_batch() -> None:
    sampler = SequenceBatchSampler(
        np.arange(40, dtype=np.int64),
        batch_size=5,
        sequence_length=6,
        seed=11,
    )
    sampler.next_batch()
    state = sampler.state_dict()
    expected = sampler.next_batch()
    restored = SequenceBatchSampler(
        np.arange(40, dtype=np.int64),
        batch_size=5,
        sequence_length=6,
        seed=11,
    )

    restored.load_state_dict(state)
    actual = restored.next_batch()

    assert all(
        np.array_equal(left, right)
        for left, right in zip(expected, actual, strict=True)
    )


def test_sequence_sampler_rejects_incompatible_or_malformed_state() -> None:
    sampler = SequenceBatchSampler(
        np.arange(10, dtype=np.int64),
        batch_size=2,
        sequence_length=3,
        seed=2,
    )
    state = sampler.state_dict()
    malformed = copy.deepcopy(state)
    malformed["sequence_length"] = 2

    with pytest.raises(ValueError, match="sequence_length"):
        sampler.load_state_dict(malformed)
    missing = copy.deepcopy(state)
    missing.pop("rng_state")
    with pytest.raises(ValueError, match="keys"):
        sampler.load_state_dict(missing)


@pytest.mark.parametrize(
    ("tokens", "batch_size", "sequence_length", "error"),
    [
        (np.ones((2, 2), dtype=np.int64), 2, 1, ValueError),
        (np.arange(5, dtype=np.float64), 2, 1, TypeError),
        (np.arange(5, dtype=np.int64), 0, 1, ValueError),
        (np.arange(5, dtype=np.int64), 2, 0, ValueError),
        (np.arange(5, dtype=np.int64), 2, 5, ValueError),
    ],
)
def test_sequence_sampler_rejects_invalid_configuration(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        SequenceBatchSampler(
            tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=0,
        )


def test_chronological_streams_do_not_create_boundary_pair() -> None:
    text = "abcabcabcabc"
    dataset = prepare_token_stream_dataset(text, train_fraction=0.5)
    train_sampler = SequenceBatchSampler(
        dataset.train_tokens,
        batch_size=1,
        sequence_length=5,
        seed=0,
    )
    validation_sampler = SequenceBatchSampler(
        dataset.validation_tokens,
        batch_size=1,
        sequence_length=5,
        seed=0,
    )

    train_inputs, train_targets = train_sampler.batch_from_starts(
        np.array([0], dtype=np.int64)
    )
    validation_inputs, validation_targets = validation_sampler.batch_from_starts(
        np.array([0], dtype=np.int64)
    )

    assert dataset.split_index == 6
    assert np.array_equal(train_inputs[0], dataset.train_tokens[:-1])
    assert np.array_equal(train_targets[0], dataset.train_tokens[1:])
    assert np.array_equal(validation_inputs[0], dataset.validation_tokens[:-1])
    assert np.array_equal(validation_targets[0], dataset.validation_tokens[1:])
