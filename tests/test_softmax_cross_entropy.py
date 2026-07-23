import numpy as np
import pytest

from localml_scholar.losses import (
    multiclass_cross_entropy,
    softmax_cross_entropy_backward,
    softmax_cross_entropy_forward,
    stable_softmax,
)


def test_softmax_probabilities_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, -1.0], [0.5, 0.5, 0.5]])

    probabilities = stable_softmax(logits)

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)


def test_softmax_is_finite_for_large_logits() -> None:
    logits = np.array([[1_000_000.0, 1_000_001.0], [-1_000_001.0, -1_000_000.0]])

    probabilities = stable_softmax(logits)

    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_cross_entropy_matches_hand_calculation() -> None:
    probabilities = np.array([[0.25, 0.75], [0.8, 0.2]])
    targets = np.array([1, 0], dtype=np.int64)
    expected = (-np.log(0.75) - np.log(0.8)) / 2.0

    actual = multiclass_cross_entropy(probabilities, targets)

    assert actual == pytest.approx(expected)


def test_combined_forward_matches_probability_cross_entropy() -> None:
    logits = np.array([[2.0, -1.0, 0.5], [-0.2, 0.7, 1.1]])
    targets = np.array([0, 2], dtype=np.int64)

    loss, probabilities = softmax_cross_entropy_forward(logits, targets)

    assert loss == pytest.approx(multiclass_cross_entropy(probabilities, targets))


def test_analytical_logits_gradient_matches_finite_differences() -> None:
    logits = np.array([[0.3, -0.2, 1.1], [-1.0, 0.4, 0.2]], dtype=np.float64)
    targets = np.array([2, 1], dtype=np.int64)
    loss, probabilities = softmax_cross_entropy_forward(logits, targets)
    assert np.isfinite(loss)
    analytical = softmax_cross_entropy_backward(probabilities, targets)
    numerical = np.zeros_like(logits)
    epsilon = 1e-6

    for index in np.ndindex(logits.shape):
        original = logits[index]
        logits[index] = original + epsilon
        plus, _ = softmax_cross_entropy_forward(logits, targets)
        logits[index] = original - epsilon
        minus, _ = softmax_cross_entropy_forward(logits, targets)
        logits[index] = original
        numerical[index] = (plus - minus) / (2.0 * epsilon)

    assert np.allclose(analytical, numerical, rtol=1e-6, atol=1e-7)


def test_malformed_loss_shapes_raise_useful_errors() -> None:
    with pytest.raises(ValueError, match=r"shape \(batch, classes\)"):
        softmax_cross_entropy_forward(np.zeros(3), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="targets must have shape"):
        softmax_cross_entropy_forward(np.zeros((2, 3)), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match=r"targets must be in \[0, 3\)"):
        softmax_cross_entropy_forward(np.zeros((1, 3)), np.array([3], dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        stable_softmax(np.array([0.0, np.inf]))
