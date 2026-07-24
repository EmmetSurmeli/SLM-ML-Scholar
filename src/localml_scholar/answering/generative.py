"""Grounded generation through the project's explicit local transformer."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from localml_scholar.answering.context import GroundedContext
from localml_scholar.generation import generate_transformer_ids
from localml_scholar.models.transformer_lm import TransformerLanguageModel
from localml_scholar.tokenizer import Tokenizer


@dataclass(frozen=True)
class GroundedGenerationConfig:
    """Fixed-length local generation and sampling policy."""

    maximum_new_tokens: int = 64
    greedy: bool = True
    temperature: float = 1.0
    top_k: int | None = None
    seed: int | None = 0
    decoded_stop_delimiter: str | None = None
    decode_errors: str = "replace"

    def __post_init__(self) -> None:
        if isinstance(self.maximum_new_tokens, bool) or not isinstance(
            self.maximum_new_tokens, int
        ):
            raise TypeError("maximum_new_tokens must be an integer.")
        if self.maximum_new_tokens <= 0:
            raise ValueError("maximum_new_tokens must be positive.")
        if not isinstance(self.greedy, bool):
            raise TypeError("greedy must be boolean.")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise TypeError("temperature must be a real number.")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        object.__setattr__(self, "temperature", float(self.temperature))
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be None or a positive integer.")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be None or a non-negative integer.")
        if self.decoded_stop_delimiter is not None and (
            not isinstance(self.decoded_stop_delimiter, str)
            or not self.decoded_stop_delimiter
        ):
            raise ValueError(
                "decoded_stop_delimiter must be None or a non-empty string."
            )
        if self.decode_errors not in {"strict", "replace"}:
            raise ValueError("decode_errors must be 'strict' or 'replace'.")

    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass(frozen=True)
class GroundedGeneration:
    """Raw local generation, conservative processing, and token trace."""

    raw_text: str
    processed_text: str
    generated_token_ids: tuple[int, ...]
    prompt_token_count: int
    stopped_on_delimiter: bool


class GroundedGenerativeAnswerer:
    """Run one explicitly supplied model/tokenizer pair on a grounded context."""

    def __init__(
        self,
        model: TransformerLanguageModel,
        tokenizer: Tokenizer,
        *,
        config: GroundedGenerationConfig | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        if not isinstance(model, TransformerLanguageModel):
            raise TypeError("model must be a TransformerLanguageModel.")
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must implement the Tokenizer interface.")
        if tokenizer.vocabulary_size != model.config.vocabulary_size:
            raise ValueError("Model and tokenizer vocabulary sizes must match.")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GroundedGenerationConfig()
        if not isinstance(self.config, GroundedGenerationConfig):
            raise TypeError("config must be GroundedGenerationConfig.")
        if checkpoint_sha256 is not None and (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in checkpoint_sha256
            )
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest.")
        if checkpoint_path is not None and (
            not isinstance(checkpoint_path, str) or not checkpoint_path
        ):
            raise ValueError("checkpoint_path must be None or a non-empty string.")
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_path = checkpoint_path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        config: GroundedGenerationConfig | None = None,
    ) -> GroundedGenerativeAnswerer:
        """Load an explicit model-only checkpoint with its matching tokenizer."""
        source = Path(path)
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Grounded-generation checkpoint does not exist: {source}"
            ) from None
        model, tokenizer = TransformerLanguageModel.load_checkpoint_with_tokenizer(
            source
        )
        return cls(
            model,
            tokenizer,
            config=config,
            checkpoint_sha256=digest,
            checkpoint_path=str(source),
        )

    def generate(self, context: GroundedContext) -> GroundedGeneration:
        """Generate only when controls, evidence, and output fit without cropping."""
        if not isinstance(context, GroundedContext):
            raise TypeError("context must be a GroundedContext.")
        if context.generation_allowance < self.config.maximum_new_tokens:
            raise ValueError(
                "Grounded context generation allowance is smaller than the "
                "configured output length."
            )
        if context.maximum_context_tokens > self.model.config.maximum_context_length:
            raise ValueError("Grounded context exceeds the model context limit.")
        prompt_ids = self.tokenizer.encode(context.prompt)
        if prompt_ids.size != context.prompt_token_count:
            raise ValueError("Grounded context token count is inconsistent.")
        if (
            prompt_ids.size + self.config.maximum_new_tokens
            > self.model.config.maximum_context_length
        ):
            raise ValueError(
                "Prompt plus generated output would crop grounded controls or evidence."
            )
        previous_mode = self.model.training
        generated = generate_transformer_ids(
            self.model,
            prompt_ids[None, :],
            max_new_tokens=self.config.maximum_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            greedy=self.config.greedy,
            seed=self.config.seed,
        )[0]
        if self.model.training != previous_mode or self.model.has_pending_cache():
            raise RuntimeError(
                "Generation did not restore the model inference lifecycle."
            )
        new_ids = generated[prompt_ids.size :]
        raw = self.tokenizer.decode(new_ids, errors=self.config.decode_errors)
        processed = raw.strip()
        stopped = False
        delimiter = self.config.decoded_stop_delimiter
        if delimiter is not None and delimiter in processed:
            processed = processed.split(delimiter, maxsplit=1)[0].rstrip()
            stopped = True
        return GroundedGeneration(
            raw_text=raw,
            processed_text=processed,
            generated_token_ids=tuple(int(value) for value in new_ids),
            prompt_token_count=int(prompt_ids.size),
            stopped_on_delimiter=stopped,
        )
