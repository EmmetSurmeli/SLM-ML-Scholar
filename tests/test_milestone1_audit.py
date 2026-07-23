import json

import numpy as np
import pytest

from experiments.train_bigram import evaluate, load_config
from localml_scholar.data import MiniBatchSampler, prepare_bigram_dataset
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.optimizers import SGD
from localml_scholar.utils import check_bigram_gradients, safe_perplexity


def test_split_builds_pairs_inside_each_side_and_excludes_boundary() -> None:
    dataset = prepare_bigram_dataset("abcdabcd", train_fraction=0.5)

    train_pairs = list(
        zip(
            dataset.tokenizer.decode(dataset.train_inputs),
            dataset.tokenizer.decode(dataset.train_targets),
            strict=True,
        )
    )
    validation_pairs = list(
        zip(
            dataset.tokenizer.decode(dataset.validation_inputs),
            dataset.tokenizer.decode(dataset.validation_targets),
            strict=True,
        )
    )

    assert dataset.split_index == 4
    assert train_pairs == [("a", "b"), ("b", "c"), ("c", "d")]
    assert validation_pairs == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ("d", "a") not in train_pairs
    assert ("d", "a") not in validation_pairs


def test_batch_averaging_occurs_exactly_once() -> None:
    model = BigramLanguageModel(3, seed=2)
    inputs = np.array([0, 1, 0], dtype=np.int64)
    targets = np.array([1, 2, 2], dtype=np.int64)
    model.loss_and_backward(inputs, targets)
    original_gradient = model.grad_weights.copy()
    model.grad_weights.fill(0.0)

    model.loss_and_backward(
        np.tile(inputs, 2),
        np.tile(targets, 2),
    )

    assert np.allclose(model.grad_weights, original_gradient)


def test_legacy_sgd_weight_decay_is_coupled_l2() -> None:
    parameter = np.array([[1.0, -2.0]], dtype=np.float64)
    gradient = np.array([[0.3, -0.4]], dtype=np.float64)
    optimizer = SGD({"weight": parameter}, learning_rate=0.2, weight_decay=0.1)
    expected = parameter.copy() - 0.2 * (gradient + 0.1 * parameter.copy())

    optimizer.step({"weight": gradient})

    assert np.allclose(parameter, expected)


def test_bigram_gradient_checker_restores_weights_bit_exactly() -> None:
    model = BigramLanguageModel(3, seed=4)
    inputs = np.array([0, 1, 2], dtype=np.int64)
    targets = np.array([1, 2, 0], dtype=np.int64)
    original = model.weights.copy()

    result = check_bigram_gradients(model, inputs, targets)

    assert result.passed
    assert np.array_equal(model.weights, original)


def test_evaluation_preserves_existing_eval_mode() -> None:
    model = BigramLanguageModel(2).eval()
    inputs = np.array([0, 1], dtype=np.int64)
    targets = np.array([1, 0], dtype=np.int64)
    sampler = MiniBatchSampler(inputs, targets, batch_size=2, seed=3)

    loss = evaluate(model, sampler, evaluation_batches=2)

    assert np.isfinite(loss)
    assert not model.training


def test_perplexity_overflow_and_nonfinite_inputs_are_explicit() -> None:
    assert safe_perplexity(float("inf")) == np.finfo(np.float64).max
    assert np.isfinite(safe_perplexity(1e9))
    assert safe_perplexity(float("-inf")) == 0.0
    with pytest.raises(ValueError, match="NaN"):
        safe_perplexity(float("nan"))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("batch_size", True, "must be an integer"),
        ("train_fraction", 1.0, "below 1.0"),
        ("learning_rate", 0.0, "greater than"),
        ("sampling_temperature", float("inf"), "finite"),
        ("checkpoint_directory", "", "non-empty string"),
    ],
)
def test_malformed_bigram_configurations_are_rejected(
    tmp_path, field: str, value: object, message: str
) -> None:
    configuration = {
        "seed": 7,
        "train_fraction": 0.9,
        "batch_size": 32,
        "learning_rate": 1.0,
        "weight_decay": 0.0,
        "num_steps": 2,
        "evaluation_interval": 1,
        "evaluation_batches": 1,
        "generation_length": 4,
        "sampling_temperature": 0.9,
        "checkpoint_directory": "outputs/test",
    }
    configuration[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_config(path)
