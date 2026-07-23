import numpy as np
import pytest

from localml_scholar.nn.embedding import Embedding
from localml_scholar.training.gradient_check import check_module_gradients


def test_embedding_forward_lookup_and_output_shape() -> None:
    layer = Embedding(4, 3, seed=0)
    table = np.arange(12, dtype=np.float64).reshape(4, 3)
    layer.weight.load_data(table)
    indices = np.array([[2, 0], [3, 1]], dtype=np.int64)

    output = layer.forward(indices)

    assert output.shape == (2, 2, 3)
    assert np.array_equal(output, table[indices])


def test_embedding_repeated_indices_accumulate_gradients() -> None:
    layer = Embedding(3, 2, seed=0)
    indices = np.array([1, 1, 2], dtype=np.int64)
    upstream = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]])
    layer.forward(indices)

    grad_indices = layer.backward(upstream)

    assert grad_indices is None
    assert np.array_equal(layer.weight.grad[0], np.array([0.0, 0.0]))
    assert np.array_equal(layer.weight.grad[1], np.array([4.0, 6.0]))
    assert np.array_equal(layer.weight.grad[2], np.array([-1.0, 5.0]))


@pytest.mark.parametrize(
    "indices, message",
    [
        (np.array([0.0, 1.0]), "integer dtype"),
        (np.array([-1, 0]), r"lie in \[0, 3\)"),
        (np.array([0, 3]), r"lie in \[0, 3\)"),
        (np.array([], dtype=np.int64), "non-empty"),
    ],
)
def test_embedding_rejects_invalid_indices(indices: np.ndarray, message: str) -> None:
    layer = Embedding(3, 2, seed=0)

    with pytest.raises((TypeError, ValueError), match=message):
        layer.forward(indices)


def test_embedding_weight_passes_gradient_check_with_repeated_ids() -> None:
    layer = Embedding(3, 2, seed=2)
    indices = np.array([[0, 1], [1, 2]], dtype=np.int64)
    upstream = np.array(
        [[[0.2, -0.7], [1.0, 0.5]], [[-0.3, 0.9], [0.4, -1.2]]],
        dtype=np.float64,
    )

    def objective(output: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(output * upstream)), upstream.copy()

    report = check_module_gradients(
        layer,
        indices,
        objective,
        check_input=False,
        check_parameters=True,
    )

    assert report.passed
    assert [result.name for result in report.tensors] == ["weight"]
