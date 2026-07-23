import numpy as np
import pytest

from localml_scholar.nn.parameter import Parameter
from localml_scholar.optim.adam import Adam
from localml_scholar.optim.momentum import Momentum
from localml_scholar.optim.sgd import SGD


def test_sgd_exact_step_and_zero_grad() -> None:
    parameter = Parameter(np.array([1.0, -2.0]))
    parameter.grad[...] = np.array([0.5, -1.0])
    optimizer = SGD((parameter,), learning_rate=0.2)

    optimizer.step()

    assert np.array_equal(parameter.data, np.array([0.9, -1.8]))
    optimizer.zero_grad()
    assert np.array_equal(parameter.grad, np.zeros(2))


def test_sgd_uses_documented_coupled_weight_decay() -> None:
    parameter = Parameter(np.array([1.0, -2.0]))
    parameter.grad[...] = np.array([0.5, -1.0])
    optimizer = SGD((parameter,), learning_rate=0.2, weight_decay=0.1)
    expected = parameter.data - 0.2 * (parameter.grad + 0.1 * parameter.data)

    optimizer.step()

    assert np.allclose(parameter.data, expected)


def test_momentum_first_several_steps_match_hand_computation() -> None:
    parameter = Parameter(np.array([1.0, -2.0]))
    optimizer = Momentum((parameter,), learning_rate=0.1, beta=0.8)
    expected_parameter = parameter.data.copy()
    expected_velocity = np.zeros_like(expected_parameter)
    gradients = (
        np.array([0.5, -1.0]),
        np.array([-0.2, 0.3]),
        np.array([1.1, -0.4]),
    )

    for gradient in gradients:
        parameter.grad[...] = gradient
        expected_velocity = 0.8 * expected_velocity + gradient
        expected_parameter = expected_parameter - 0.1 * expected_velocity
        optimizer.step()
        assert np.allclose(parameter.data, expected_parameter)


def test_adam_first_several_steps_match_hand_computation() -> None:
    parameter = Parameter(np.array([1.0, -2.0]))
    optimizer = Adam(
        (parameter,),
        learning_rate=0.05,
        beta1=0.7,
        beta2=0.8,
        epsilon=1e-6,
    )
    expected_parameter = parameter.data.copy()
    first = np.zeros_like(expected_parameter)
    second = np.zeros_like(expected_parameter)
    gradients = (
        np.array([0.5, -1.0]),
        np.array([-0.2, 0.3]),
        np.array([1.1, -0.4]),
    )

    for step, gradient in enumerate(gradients, start=1):
        parameter.grad[...] = gradient
        first = 0.7 * first + 0.3 * gradient
        second = 0.8 * second + 0.2 * gradient**2
        first_hat = first / (1.0 - 0.7**step)
        second_hat = second / (1.0 - 0.8**step)
        expected_parameter -= 0.05 * first_hat / (np.sqrt(second_hat) + 1e-6)
        optimizer.step()
        assert np.allclose(parameter.data, expected_parameter, atol=1e-14)
        assert optimizer.step_count == step


def test_adam_maintains_independent_state_per_parameter() -> None:
    first = Parameter(np.array([1.0]))
    second = Parameter(np.array([1.0]))
    first.grad[...] = 1.0
    second.grad[...] = 2.0
    optimizer = Adam(
        (first, second),
        learning_rate=0.1,
        beta1=0.0,
        beta2=0.5,
        epsilon=1e-8,
    )

    optimizer.step()
    first.grad[...] = 0.0
    second.grad[...] = 3.0
    optimizer.step()

    assert first.data[0] != second.data[0]


def test_adam_checkpoint_restores_moments_and_step_count(tmp_path) -> None:
    original_parameter = Parameter(np.array([1.0, -1.0]))
    original = Adam((original_parameter,), learning_rate=0.02, beta1=0.8, beta2=0.9)
    original_parameter.grad[...] = np.array([0.3, -0.5])
    original.step()
    checkpoint = tmp_path / "adam.npz"
    original.save_checkpoint(checkpoint)

    restored_parameter = Parameter(original_parameter.data.copy())
    restored = Adam((restored_parameter,), learning_rate=0.02, beta1=0.8, beta2=0.9)
    restored.load_checkpoint(checkpoint)
    next_gradient = np.array([-0.7, 0.2])
    original_parameter.grad[...] = next_gradient
    restored_parameter.grad[...] = next_gradient
    original.step()
    restored.step()

    assert restored.step_count == original.step_count
    assert np.array_equal(restored_parameter.data, original_parameter.data)


def test_optimizer_rejects_duplicate_and_nonfinite_gradients() -> None:
    parameter = Parameter(np.array([1.0]))
    with pytest.raises(ValueError, match="more than once"):
        SGD((parameter, parameter), learning_rate=0.1)

    optimizer = SGD((parameter,), learning_rate=0.1)
    parameter.grad[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.step()
