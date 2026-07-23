import numpy as np
import pytest

from localml_scholar.generation import generate_text
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.tokenizer import CharacterTokenizer


def test_generation_is_deterministic_with_fixed_seed() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    model = BigramLanguageModel(tokenizer.vocabulary_size, seed=41)

    first = generate_text(
        model,
        tokenizer,
        max_new_tokens=40,
        seed=9,
        seed_text="ab",
        temperature=0.8,
    )
    second = generate_text(
        model,
        tokenizer,
        max_new_tokens=40,
        seed=9,
        seed_text="ab",
        temperature=0.8,
    )

    assert first == second
    assert len(first) == 42


def test_greedy_generation_follows_largest_logit() -> None:
    tokenizer = CharacterTokenizer.from_text("ab")
    model = BigramLanguageModel(tokenizer.vocabulary_size, initialization_scale=0.0)
    model.weights[...] = np.array([[0.0, 2.0], [3.0, -1.0]])

    generated = generate_text(
        model,
        tokenizer,
        max_new_tokens=4,
        start_token="a",
        greedy=True,
    )

    assert generated == "ababa"


def test_generation_rejects_unknown_seed_text() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    model = BigramLanguageModel(tokenizer.vocabulary_size)

    with pytest.raises(ValueError, match="Unknown character"):
        generate_text(model, tokenizer, max_new_tokens=1, seed_text="az")


def test_generation_validates_options() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    model = BigramLanguageModel(tokenizer.vocabulary_size)

    with pytest.raises(ValueError, match="at most one"):
        generate_text(
            model,
            tokenizer,
            max_new_tokens=1,
            seed_text="a",
            start_token="a",
        )
    with pytest.raises(ValueError, match="positive"):
        generate_text(
            model,
            tokenizer,
            max_new_tokens=1,
            seed_text="a",
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        generate_text(model, tokenizer, max_new_tokens=-1, seed_text="a")
