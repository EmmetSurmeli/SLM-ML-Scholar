from typing import Any

import numpy as np
import pytest

from localml_scholar.generation import (
    generate_transformer_ids,
    transformer_sampling_probabilities,
)
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)


def _model(**overrides: Any) -> TransformerLanguageModel:
    values: dict[str, Any] = {
        "vocabulary_size": 5,
        "maximum_context_length": 3,
        "model_dimension": 3,
        "number_of_layers": 1,
        "key_dimension": 2,
        "value_dimension": 2,
        "feed_forward_dimension": 5,
        "dtype": np.float64,
        "seed": 13,
    }
    values.update(overrides)
    return TransformerLanguageModel(TransformerConfig(**values))


def _bias_only_model() -> TransformerLanguageModel:
    model = _model()
    model.language_model_head.weight.data.fill(0.0)
    assert model.language_model_head.bias is not None
    model.language_model_head.bias.data[...] = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return model


def test_transformer_sampling_probabilities_filter_exactly_and_normalize() -> None:
    logits = np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64)

    probabilities = transformer_sampling_probabilities(
        logits,
        temperature=0.5,
        top_k=2,
    )

    assert np.sum(probabilities, axis=-1)[0] == pytest.approx(1.0)
    assert np.all(probabilities >= 0.0)
    assert np.array_equal(probabilities[:, 2:], np.zeros((1, 2)))


def test_top_k_ties_use_low_vocabulary_index_first() -> None:
    logits = np.ones((1, 4), dtype=np.float64)

    probabilities = transformer_sampling_probabilities(logits, top_k=2)

    assert np.array_equal(probabilities, np.array([[0.5, 0.5, 0.0, 0.0]]))


def test_greedy_generation_preserves_prompt_batch_and_restores_mode() -> None:
    model = _bias_only_model()
    prompt = np.array([[0, 1], [2, 3]], dtype=np.int64)

    generated = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=4,
        greedy=True,
        seed=999,
    )

    assert generated.shape == (2, 6)
    assert np.array_equal(generated[:, :2], prompt)
    assert np.array_equal(generated[:, 2:], np.full((2, 4), 4))
    assert model.training
    assert not model.has_pending_cache()


def test_prompt_longer_than_context_is_explicitly_cropped_during_generation() -> None:
    model = _bias_only_model()
    prompt = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)

    generated = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=2,
        greedy=True,
    )

    assert generated.shape == (1, 7)
    assert np.array_equal(generated[0, -2:], np.array([4, 4]))


def test_seeded_sampling_is_deterministic_and_can_diverge() -> None:
    model = _model(vocabulary_size=7)
    prompt = np.array([[0]], dtype=np.int64)

    first = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=20,
        seed=3,
    )
    second = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=20,
        seed=3,
    )
    different = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=20,
        seed=4,
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all((first >= 0) & (first < model.config.vocabulary_size))


def test_top_one_sampling_matches_greedy() -> None:
    model = _model()
    prompt = np.array([[0, 1]], dtype=np.int64)

    greedy = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=8,
        greedy=True,
    )
    top_one = generate_transformer_ids(
        model,
        prompt,
        max_new_tokens=8,
        top_k=1,
        seed=123,
    )

    assert np.array_equal(top_one, greedy)


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_generation_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        generate_transformer_ids(
            _model(),
            np.array([[0]], dtype=np.int64),
            max_new_tokens=1,
            temperature=temperature,
        )


@pytest.mark.parametrize("top_k", [0, 6])
def test_generation_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        generate_transformer_ids(
            _model(),
            np.array([[0]], dtype=np.int64),
            max_new_tokens=1,
            top_k=top_k,
        )


def test_generation_rejects_invalid_prompt_and_length() -> None:
    model = _model()
    with pytest.raises(TypeError, match="integer"):
        generate_transformer_ids(
            model,
            np.array([[0.0]]),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="input_ids"):
        generate_transformer_ids(
            model,
            np.array([[5]], dtype=np.int64),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        generate_transformer_ids(
            model,
            np.array([[0]], dtype=np.int64),
            max_new_tokens=-1,
        )


def test_generation_rejects_pending_training_forward() -> None:
    model = _model()
    prompt = np.array([[0, 1]], dtype=np.int64)
    logits = model.forward(prompt)

    with pytest.raises(RuntimeError, match="inference mode.*cache"):
        generate_transformer_ids(
            model,
            prompt,
            max_new_tokens=1,
        )

    model.backward(np.ones_like(logits))
