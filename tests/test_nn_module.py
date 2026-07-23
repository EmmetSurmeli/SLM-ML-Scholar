import numpy as np
import pytest

from localml_scholar.models.mlp import MLP
from localml_scholar.nn.containers import Sequential
from localml_scholar.nn.linear import Linear
from localml_scholar.nn.parameter import Parameter


def test_parameter_owns_same_dtype_gradient_buffer() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)
    parameter = Parameter(source, name="example")
    source[0] = 99.0

    assert parameter.dtype == np.float32
    assert parameter.shape == (2,)
    assert parameter.size == 2
    assert parameter.data[0] == 1.0
    assert parameter.grad.dtype == parameter.data.dtype
    assert np.array_equal(parameter.grad, np.zeros(2, dtype=np.float32))


def test_parameter_rejects_integer_data_and_bad_loaded_dtype() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        Parameter(np.array([1, 2], dtype=np.int64))

    parameter = Parameter(np.array([1.0], dtype=np.float32))
    with pytest.raises(TypeError, match="does not match"):
        parameter.load_data(np.array([1.0], dtype=np.float64))


def test_foundation_seed_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Linear(2, 2, seed=-1)
    with pytest.raises(ValueError, match="non-negative"):
        MLP(2, 2, 2, seed=-1)


def test_named_parameter_traversal_is_nested_and_deterministic() -> None:
    model = MLP(2, 3, 2, seed=5)

    names = [name for name, _ in model.named_parameters()]

    assert names == [
        "network.0.weight",
        "network.0.bias",
        "network.2.weight",
        "network.2.bias",
    ]
    assert model.parameters() == tuple(
        parameter for _, parameter in model.named_parameters()
    )


def test_train_eval_and_zero_grad_propagate_recursively() -> None:
    model = MLP(2, 3, 2, seed=4)
    for parameter in model.parameters():
        parameter.grad.fill(2.0)

    model.eval()

    assert not model.training
    assert all(not module.training for module in model.network)
    model.zero_grad()
    assert all(
        np.count_nonzero(parameter.grad) == 0 for parameter in model.parameters()
    )
    model.train()
    assert model.training
    assert all(module.training for module in model.network)


def test_repeated_training_forward_is_rejected_until_backward() -> None:
    layer = Linear(2, 2, seed=1)
    inputs = np.ones((1, 2), dtype=np.float64)

    layer.forward(inputs)

    with pytest.raises(RuntimeError, match="cannot run twice"):
        layer.forward(inputs)
    layer.clear_cache()
    with pytest.raises(RuntimeError, match="requires one unmatched"):
        layer.backward(np.ones((1, 2), dtype=np.float64))


def test_eval_forward_does_not_create_backward_cache() -> None:
    layer = Linear(2, 2, seed=1).eval()
    inputs = np.ones((1, 2), dtype=np.float64)

    layer.forward(inputs)
    layer.forward(inputs)

    with pytest.raises(RuntimeError, match="requires one unmatched"):
        layer.backward(np.ones((1, 2), dtype=np.float64))


def test_mode_change_with_pending_cache_is_rejected() -> None:
    layer = Linear(2, 2, seed=1)
    layer.forward(np.ones((1, 2), dtype=np.float64))

    with pytest.raises(RuntimeError, match="mode.*cache"):
        layer.eval()


def test_sequential_rejects_reused_module_instance() -> None:
    layer = Linear(2, 2, seed=1)

    with pytest.raises(ValueError, match="cannot reuse"):
        Sequential((layer, layer))
