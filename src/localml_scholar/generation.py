"""Autoregressive text generation without external sampling libraries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from localml_scholar.losses import stable_softmax
from localml_scholar.models.bigram import BigramLanguageModel
from localml_scholar.models.transformer_lm import TransformerLanguageModel
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


def transformer_sampling_probabilities(
    logits: NDArray[np.floating],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> NDArray[np.floating]:
    """Return stable final-axis sampling probabilities after top-k filtering.

    Ties are resolved by stable vocabulary-index order. Tokens outside the
    selected set receive an exact probability of zero.
    """
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            "logits must have shape (batch, vocabulary) with positive dimensions."
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"logits must have a floating-point dtype, got {values.dtype}.")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"logits must use float32 or float64, got {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must contain only finite values.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number.")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    vocabulary_size = values.shape[-1]
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be None or an integer.")
        if not 1 <= top_k <= vocabulary_size:
            raise ValueError(f"top_k must lie in [1, {vocabulary_size}].")

    scaled = values / np.asarray(temperature, dtype=values.dtype)
    if top_k is None or top_k == vocabulary_size:
        return stable_softmax(scaled)

    selected = np.argsort(-scaled, axis=-1, kind="stable")[:, :top_k]
    selected_logits = np.take_along_axis(scaled, selected, axis=-1)
    selected_probabilities = stable_softmax(selected_logits)
    probabilities = np.zeros_like(scaled)
    np.put_along_axis(
        probabilities,
        selected,
        selected_probabilities,
        axis=-1,
    )
    return probabilities


def generate_transformer_ids(
    model: TransformerLanguageModel,
    input_ids: NDArray[np.integer],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    seed: int | None = None,
) -> NDArray[np.int64]:
    """Generate token IDs by recomputing the cropped context at every step.

    ``temperature`` and ``top_k`` are validated in greedy mode but do not alter
    the argmax result. A zero generation length returns a copied prompt.
    """
    if not isinstance(model, TransformerLanguageModel):
        raise TypeError("model must be a TransformerLanguageModel.")
    prompt = np.asarray(input_ids)
    if not np.issubdtype(prompt.dtype, np.integer):
        raise TypeError(f"input_ids must have an integer dtype, got {prompt.dtype}.")
    if prompt.ndim != 2:
        raise ValueError(
            "input_ids must have exactly two dimensions (B, T), "
            f"got shape {prompt.shape}."
        )
    if prompt.shape[0] == 0 or prompt.shape[1] == 0:
        raise ValueError("input_ids batch and prompt dimensions must be positive.")
    if np.any(prompt < 0) or np.any(prompt >= model.config.vocabulary_size):
        raise ValueError(f"input_ids must lie in [0, {model.config.vocabulary_size}).")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if not isinstance(greedy, bool):
        raise TypeError("greedy must be a boolean.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be None or an integer.")
    if seed is not None and seed < 0:
        raise ValueError("seed must be non-negative.")

    # Validate sampling configuration before a possible zero-length return.
    transformer_sampling_probabilities(
        np.zeros(
            (1, model.config.vocabulary_size),
            dtype=model.dtype,
        ),
        temperature=temperature,
        top_k=top_k,
    )
    generated = np.array(prompt, dtype=np.int64, copy=True)
    if max_new_tokens == 0:
        return generated

    rng = np.random.default_rng(seed)
    with model.inference_mode():
        for _ in range(max_new_tokens):
            context = generated[:, -model.config.maximum_context_length :]
            logits = model.forward(context)[:, -1, :]
            if greedy:
                next_ids = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
            else:
                probabilities = transformer_sampling_probabilities(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                )
                next_ids = np.fromiter(
                    (
                        rng.choice(model.config.vocabulary_size, p=row)
                        for row in probabilities
                    ),
                    dtype=np.int64,
                    count=generated.shape[0],
                )
            generated = np.concatenate((generated, next_ids[:, None]), axis=1)
    return generated
