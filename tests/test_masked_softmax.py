import numpy as np
import pytest

from localml_scholar.nn.attention import (
    masked_softmax,
    masked_softmax_backward,
)
from localml_scholar.nn.masks import causal_attention_mask


def test_masked_softmax_rows_sum_to_one_and_blocked_entries_are_zero() -> None:
    logits = np.array(
        [
            [[2.0, -1.0, 4.0], [0.5, 1.5, -2.0], [3.0, 2.0, 1.0]],
            [[-2.0, 7.0, 1.0], [3.5, -0.5, 9.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    mask = causal_attention_mask(3)

    probabilities = masked_softmax(logits, mask)
    broadcast_mask = np.broadcast_to(mask, logits.shape)

    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.all(probabilities[~broadcast_mask] == 0.0)
    assert np.all(probabilities[broadcast_mask] > 0.0)


def test_masked_softmax_is_finite_for_extreme_logits() -> None:
    maximum = np.finfo(np.float64).max
    logits = np.array([[[maximum, -maximum], [-maximum, maximum]]], dtype=np.float64)

    with np.errstate(all="raise"):
        probabilities = masked_softmax(logits, causal_attention_mask(2))

    assert np.array_equal(
        probabilities,
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    )
    assert np.all(np.isfinite(probabilities))


def test_single_allowed_entry_has_probability_one() -> None:
    logits = np.array([[[123.0, -456.0, 999.0]]], dtype=np.float64)
    mask = np.array([[[False, True, False]]])

    probabilities = masked_softmax(logits, mask)

    assert np.array_equal(probabilities, np.array([[[0.0, 1.0, 0.0]]]))


def test_masked_softmax_preserves_float32() -> None:
    logits = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    mask = causal_attention_mask(2)
    probabilities = masked_softmax(logits, mask)
    gradient = masked_softmax_backward(probabilities, np.ones_like(probabilities), mask)

    assert probabilities.dtype == np.float32
    assert gradient.dtype == np.float32


def test_masked_softmax_backward_matches_all_finite_differences() -> None:
    logits = np.array(
        [[[0.3, -1.1, 2.0], [0.7, -0.4, 1.3], [-0.2, 0.5, 0.9]]],
        dtype=np.float64,
    )
    mask = causal_attention_mask(3)
    upstream = np.array(
        [[[0.4, -0.8, 1.2], [-0.3, 0.9, 0.2], [1.1, -0.6, 0.7]]],
        dtype=np.float64,
    )
    probabilities = masked_softmax(logits, mask)
    analytical = masked_softmax_backward(probabilities, upstream, mask)
    numerical = np.zeros_like(logits)
    epsilon = 1e-6

    for index in np.ndindex(logits.shape):
        original = logits[index]
        logits[index] = original + epsilon
        plus = float(np.sum(masked_softmax(logits, mask) * upstream))
        logits[index] = original - epsilon
        minus = float(np.sum(masked_softmax(logits, mask) * upstream))
        logits[index] = original
        numerical[index] = (plus - minus) / (2.0 * epsilon)

    broadcast_mask = np.broadcast_to(mask, logits.shape)
    assert np.allclose(analytical, numerical, rtol=1e-6, atol=1e-8)
    assert np.all(analytical[~broadcast_mask] == 0.0)
    assert np.all(numerical[~broadcast_mask] == 0.0)


def test_masked_softmax_rejects_all_masked_rows() -> None:
    logits = np.zeros((1, 2, 2), dtype=np.float64)
    mask = np.array([[[True, False], [False, False]]])

    with pytest.raises(ValueError, match="at least one allowed"):
        masked_softmax(logits, mask)


def test_masked_softmax_rejects_malformed_masks_and_gradients() -> None:
    logits = np.zeros((1, 2, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="boolean"):
        masked_softmax(logits, np.ones((1, 2, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="cannot broadcast"):
        masked_softmax(logits, np.ones((3, 3), dtype=np.bool_))

    probabilities = masked_softmax(logits, causal_attention_mask(2))
    with pytest.raises(ValueError, match="must equal"):
        masked_softmax_backward(
            probabilities,
            np.ones((1, 2, 1), dtype=np.float64),
            causal_attention_mask(2),
        )
