"""Autoregressive text generation without external sampling libraries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localml_scholar.losses import stable_softmax
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.tokenizer import CharacterTokenizer

FloatArray = NDArray[np.float64]


def sample_next_token(
    logits: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    greedy: bool = False,
) -> int:
    """Choose one token from a one-dimensional logit vector."""
    values = np.asarray(logits)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"logits must be a non-empty 1D array, got {values.shape}.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"logits must have a floating-point dtype, got {values.dtype}.")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"logits must use float32 or float64, got {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number.")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    if not isinstance(greedy, bool):
        raise TypeError("greedy must be a boolean.")

    if greedy:
        return int(np.argmax(values))
    probabilities: FloatArray = stable_softmax(values / float(temperature))
    return int(rng.choice(values.size, p=probabilities))


def generate_text(
    model: BigramLanguageModel,
    tokenizer: CharacterTokenizer,
    *,
    max_new_tokens: int,
    seed: int = 0,
    temperature: float = 1.0,
    greedy: bool = False,
    seed_text: str | None = None,
    start_token: str | None = None,
) -> str:
    """Generate text autoregressively, including the supplied prefix."""
    if model.vocabulary_size != tokenizer.vocabulary_size:
        raise ValueError(
            "Model and tokenizer vocabulary sizes differ: "
            f"{model.vocabulary_size} versus {tokenizer.vocabulary_size}."
        )
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed_text is not None and start_token is not None:
        raise ValueError("Provide at most one of seed_text and start_token.")

    if seed_text is not None:
        if not seed_text:
            raise ValueError("seed_text must not be empty.")
        prefix = seed_text
    elif start_token is not None:
        if len(start_token) != 1:
            raise ValueError("start_token must be exactly one character.")
        prefix = start_token
    else:
        prefix = tokenizer.characters[0]

    generated = tokenizer.encode(prefix).tolist()
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        current = np.asarray([generated[-1]], dtype=np.int64)
        logits = model.forward(current)[0]
        next_token = sample_next_token(
            logits, rng, temperature=temperature, greedy=greedy
        )
        generated.append(next_token)
    return tokenizer.decode(np.asarray(generated, dtype=np.int64))
