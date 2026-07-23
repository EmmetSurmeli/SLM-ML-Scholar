import numpy as np
import pytest

from localml_scholar.nn.parameter import Parameter
from localml_scholar.training.clipping import (
    clip_grad_norm,
    global_gradient_norm,
)


def test_global_norm_and_no_clipping_below_threshold() -> None:
    first = Parameter(np.array([0.0, 0.0]))
    second = Parameter(np.array([0.0]))
    first.grad[...] = np.array([3.0, 0.0])
    second.grad[...] = np.array([4.0])
    before = [first.grad.copy(), second.grad.copy()]

    norm = clip_grad_norm((first, second), max_norm=6.0)

    assert norm == 5.0
    assert np.array_equal(first.grad, before[0])
    assert np.array_equal(second.grad, before[1])


def test_clipping_scales_all_gradients_uniformly() -> None:
    first = Parameter(np.array([0.0, 0.0]))
    second = Parameter(np.array([0.0]))
    first.grad[...] = np.array([3.0, 0.0])
    second.grad[...] = np.array([4.0])

    norm = clip_grad_norm((first, second), max_norm=2.5, epsilon=1e-12)
    expected_scale = 2.5 / (5.0 + 1e-12)

    assert norm == 5.0
    assert np.allclose(first.grad, np.array([3.0, 0.0]) * expected_scale)
    assert np.allclose(second.grad, np.array([4.0]) * expected_scale)
    assert global_gradient_norm((first, second)) == pytest.approx(5.0 * expected_scale)


def test_zero_gradients_have_zero_norm_and_are_unchanged() -> None:
    parameter = Parameter(np.zeros(3))

    norm = clip_grad_norm((parameter,), max_norm=1.0)

    assert norm == 0.0
    assert np.array_equal(parameter.grad, np.zeros(3))


def test_nonfinite_gradient_is_rejected() -> None:
    parameter = Parameter(np.zeros(2))
    parameter.grad[1] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        global_gradient_norm((parameter,))


def test_scaled_norm_avoids_overflow_for_large_finite_components() -> None:
    parameter = Parameter(np.zeros(2))
    parameter.grad[...] = np.array([1e300, 1e300])

    norm = global_gradient_norm((parameter,))

    assert np.isfinite(norm)
    assert norm == pytest.approx(np.sqrt(2.0) * 1e300)
