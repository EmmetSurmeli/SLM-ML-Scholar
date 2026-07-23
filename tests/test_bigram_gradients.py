import json

import numpy as np
import pytest

from experiments.train_bigram import load_config
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.optimizers import SGD
from localml_scholar.utils import check_bigram_gradients


def test_bigram_weight_gradient_passes_all_parameter_check() -> None:
    model = BigramLanguageModel(vocabulary_size=3, seed=11)
    inputs = np.array([0, 1, 0, 2], dtype=np.int64)
    targets = np.array([1, 2, 2, 0], dtype=np.int64)

    result = check_bigram_gradients(
        model,
        inputs,
        targets,
        epsilon=1e-5,
        tolerance=1e-6,
    )

    assert result.passed
    assert result.checked_parameters == 9
    assert result.maximum_relative_error < 1e-6


def test_subset_gradient_check_is_deterministic() -> None:
    model = BigramLanguageModel(vocabulary_size=5, seed=3)
    inputs = np.array([0, 1, 3], dtype=np.int64)
    targets = np.array([1, 3, 4], dtype=np.int64)

    first = check_bigram_gradients(model, inputs, targets, max_checks=6, seed=19)
    second = check_bigram_gradients(model, inputs, targets, max_checks=6, seed=19)

    assert first == second
    assert first.checked_parameters == 6


def test_one_sgd_step_decreases_loss() -> None:
    model = BigramLanguageModel(vocabulary_size=2, seed=0, initialization_scale=0.0)
    optimizer = SGD(model.parameters(), learning_rate=0.5)
    inputs = np.array([0, 0, 1, 1], dtype=np.int64)
    targets = np.array([1, 1, 0, 0], dtype=np.int64)
    before = model.loss(inputs, targets)

    optimizer.zero_grad(model.gradients())
    model.loss_and_backward(inputs, targets)
    optimizer.step(model.gradients())
    after = model.loss(inputs, targets)

    assert after < before


def test_repeated_input_rows_accumulate_gradients() -> None:
    model = BigramLanguageModel(vocabulary_size=2, seed=0, initialization_scale=0.0)
    inputs = np.array([0, 0], dtype=np.int64)
    targets = np.array([1, 1], dtype=np.int64)

    model.loss_and_backward(inputs, targets)

    assert np.allclose(model.grad_weights[0], np.array([0.5, -0.5]))
    assert np.allclose(model.grad_weights[1], 0.0)


def test_checkpoint_round_trip_preserves_outputs_exactly(tmp_path) -> None:
    model = BigramLanguageModel(vocabulary_size=4, seed=23)
    inputs = np.array([3, 0, 2], dtype=np.int64)
    expected = model.forward(inputs).copy()
    checkpoint = tmp_path / "model.npz"

    model.save_checkpoint(checkpoint)
    loaded = BigramLanguageModel.load_checkpoint(checkpoint)

    assert np.array_equal(loaded.forward(inputs), expected)
    assert loaded.parameter_count == model.parameter_count
    assert loaded.configuration == model.configuration
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert "model_config_json" in saved.files


def test_optimizer_rejects_bad_gradient_shape_and_nonfinite_values() -> None:
    model = BigramLanguageModel(vocabulary_size=2)
    optimizer = SGD(model.parameters(), learning_rate=0.1)

    with pytest.raises(ValueError, match="shape"):
        optimizer.step({"weights": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="non-finite"):
        optimizer.step({"weights": np.full((2, 2), np.nan)})
    with pytest.raises(TypeError, match="does not match"):
        optimizer.step({"weights": np.zeros((2, 2), dtype=np.float32)})


def test_model_rejects_malformed_token_and_gradient_shapes() -> None:
    model = BigramLanguageModel(vocabulary_size=3)
    with pytest.raises(ValueError, match="one-dimensional"):
        model.forward(np.zeros((1, 1), dtype=np.int64))
    with pytest.raises(ValueError, match=r"lie in \[0, 3\)"):
        model.forward(np.array([3], dtype=np.int64))
    with pytest.raises(ValueError, match="grad_logits must have shape"):
        model.backward(np.array([0], dtype=np.int64), np.zeros((1, 2)))


def test_invalid_training_config_raises_useful_error(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"seed": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_config(path)
