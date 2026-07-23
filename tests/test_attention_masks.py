import numpy as np
import pytest

from localml_scholar.nn.masks import causal_attention_mask


@pytest.mark.parametrize("sequence_length", [1, 2, 5])
def test_causal_mask_has_expected_allowed_entries(sequence_length: int) -> None:
    mask = causal_attention_mask(sequence_length)
    expected = np.tril(np.ones((sequence_length, sequence_length), dtype=np.bool_))[
        None, :, :
    ]

    assert mask.shape == (1, sequence_length, sequence_length)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, expected)
    assert not mask.flags.writeable


def test_causal_mask_allows_past_and_diagonal_but_blocks_future() -> None:
    mask = causal_attention_mask(4)[0]

    assert np.all(np.diag(mask))
    assert mask[3, 0]
    assert mask[2, 1]
    assert not mask[0, 1]
    assert not mask[1, 3]


@pytest.mark.parametrize("value", [0, -1])
def test_causal_mask_rejects_nonpositive_lengths(value: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        causal_attention_mask(value)


def test_causal_mask_rejects_noninteger_length() -> None:
    with pytest.raises(TypeError, match="integer"):
        causal_attention_mask(2.5)  # type: ignore[arg-type]


def test_causal_mask_broadcasts_across_batch_scores() -> None:
    mask = causal_attention_mask(3)
    scores = np.zeros((4, 3, 3), dtype=np.float64)

    broadcast = np.broadcast_to(mask, scores.shape)

    assert broadcast.shape == scores.shape
    assert np.array_equal(broadcast[0], broadcast[3])
