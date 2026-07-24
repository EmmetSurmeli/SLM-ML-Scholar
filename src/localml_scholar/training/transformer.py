"""Deterministic training, evaluation, and resumption for the manual transformer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from localml_scholar._version import __version__
from localml_scholar.data import SequenceBatchSampler
from localml_scholar.losses import softmax_cross_entropy_loss_and_gradient
from localml_scholar.models.transformer_lm import (
    TransformerConfig,
    TransformerLanguageModel,
)
from localml_scholar.optim.adam import Adam
from localml_scholar.optim.base import Optimizer
from localml_scholar.optim.momentum import Momentum
from localml_scholar.optim.sgd import SGD
from localml_scholar.serialization import atomic_savez
from localml_scholar.tokenizer import CharacterTokenizer
from localml_scholar.training.clipping import (
    clip_grad_norm,
    global_gradient_norm,
)
from localml_scholar.training.config import TransformerTrainingConfig
from localml_scholar.utils import safe_perplexity


def _token_stream_digest(values: NDArray[np.int64]) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _validated_token_stream(
    values: NDArray[np.integer],
    *,
    name: str,
    vocabulary_size: int,
) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(array < 0) or np.any(array >= vocabulary_size):
        raise ValueError(f"{name} contains token IDs outside the model vocabulary.")
    return np.array(array, dtype=np.int64, copy=True)


def build_optimizer(
    model: TransformerLanguageModel,
    config: TransformerTrainingConfig,
) -> Optimizer:
    """Construct the configured optimizer over deterministic model parameters."""
    if config.optimizer_name == "sgd":
        return SGD(model.parameters(), learning_rate=config.learning_rate)
    if config.optimizer_name == "momentum":
        return Momentum(
            model.parameters(),
            learning_rate=config.learning_rate,
            beta=config.momentum_beta,
        )
    return Adam(
        model.parameters(),
        learning_rate=config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.optimizer_epsilon,
    )


@dataclass(frozen=True)
class TrainingStepMetrics:
    """Metrics produced by one completed parameter update."""

    step: int
    loss: float
    perplexity: float
    pre_clipping_gradient_norm: float
    post_clipping_gradient_norm: float
    learning_rate: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Token-weighted evaluation loss and perplexity."""

    loss: float
    perplexity: float
    predicted_tokens: int
    batches: int


def evaluate_language_model(
    model: TransformerLanguageModel,
    token_ids: NDArray[np.integer],
    *,
    batch_size: int,
    sequence_length: int,
    batches: int,
    seed: int,
) -> EvaluationMetrics:
    """Evaluate fixed seeded batches without caches, gradients, or RNG side effects."""
    if not isinstance(model, TransformerLanguageModel):
        raise TypeError("model must be a TransformerLanguageModel.")
    if isinstance(batches, bool) or not isinstance(batches, int):
        raise TypeError("batches must be an integer.")
    if batches <= 0:
        raise ValueError("batches must be positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if sequence_length > model.config.maximum_context_length:
        raise ValueError(
            f"sequence_length {sequence_length} exceeds model context length "
            f"{model.config.maximum_context_length}."
        )
    stream = _validated_token_stream(
        token_ids,
        name="token_ids",
        vocabulary_size=model.config.vocabulary_size,
    )
    sampler = SequenceBatchSampler(
        stream,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    total_negative_log_likelihood = 0.0
    total_tokens = 0
    with model.inference_mode():
        for _ in range(batches):
            inputs, targets = sampler.next_batch()
            logits = model.forward(inputs)
            loss, _ = softmax_cross_entropy_loss_and_gradient(logits, targets)
            predicted_tokens = int(targets.size)
            total_negative_log_likelihood += loss * predicted_tokens
            total_tokens += predicted_tokens
    mean_loss = total_negative_log_likelihood / total_tokens
    return EvaluationMetrics(
        loss=float(mean_loss),
        perplexity=safe_perplexity(mean_loss),
        predicted_tokens=total_tokens,
        batches=batches,
    )


class TransformerTrainer:
    """Own deterministic sampler, optimizer, metrics, and full training state."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        model: TransformerLanguageModel,
        tokenizer: CharacterTokenizer,
        train_tokens: NDArray[np.integer],
        validation_tokens: NDArray[np.integer],
        config: TransformerTrainingConfig,
        *,
        optimizer: Optimizer | None = None,
    ) -> None:
        if not isinstance(model, TransformerLanguageModel):
            raise TypeError("model must be a TransformerLanguageModel.")
        if not isinstance(tokenizer, CharacterTokenizer):
            raise TypeError("tokenizer must be a CharacterTokenizer.")
        if not isinstance(config, TransformerTrainingConfig):
            raise TypeError("config must be a TransformerTrainingConfig.")
        if tokenizer.vocabulary_size != model.config.vocabulary_size:
            raise ValueError("Tokenizer and model vocabulary sizes must match exactly.")
        config.validate_for_context(model.config.maximum_context_length)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_tokens = _validated_token_stream(
            train_tokens,
            name="train_tokens",
            vocabulary_size=model.config.vocabulary_size,
        )
        self.validation_tokens = _validated_token_stream(
            validation_tokens,
            name="validation_tokens",
            vocabulary_size=model.config.vocabulary_size,
        )
        self.train_sampler = SequenceBatchSampler(
            self.train_tokens,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            seed=config.seed + 1,
        )
        # Validate validation capacity at construction rather than during a run.
        SequenceBatchSampler(
            self.validation_tokens,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            seed=config.seed + 2,
        )
        resolved_optimizer = (
            build_optimizer(model, config) if optimizer is None else optimizer
        )
        if not isinstance(resolved_optimizer, Optimizer):
            raise TypeError("optimizer must be an Optimizer.")
        if resolved_optimizer.parameters != model.parameters():
            raise ValueError(
                "Optimizer parameters must exactly match model parameter order."
            )
        expected_type = {
            "sgd": SGD,
            "momentum": Momentum,
            "adam": Adam,
        }[config.optimizer_name]
        if not isinstance(resolved_optimizer, expected_type):
            raise ValueError(
                f"Training configuration requests {config.optimizer_name!r}, "
                f"got {type(resolved_optimizer).__name__}."
            )
        if resolved_optimizer.learning_rate != config.learning_rate:
            raise ValueError(
                "Optimizer learning rate does not match training configuration."
            )
        self.optimizer = resolved_optimizer
        self.completed_steps = 0
        self.best_validation_loss: float | None = None
        self.best_validation_step: int | None = None
        self.history: list[dict[str, Any]] = []

    @property
    def corpus_metadata(self) -> dict[str, Any]:
        """Return identity metadata for both isolated token streams."""
        return {
            "train_token_count": int(self.train_tokens.size),
            "train_sha256": _token_stream_digest(self.train_tokens),
            "validation_token_count": int(self.validation_tokens.size),
            "validation_sha256": _token_stream_digest(self.validation_tokens),
        }

    def train_step(self) -> TrainingStepMetrics:
        """Run one complete explicit forward/backward/update cycle."""
        if self.completed_steps >= self.config.maximum_steps:
            raise RuntimeError(
                f"Training already reached maximum_steps={self.config.maximum_steps}."
            )
        if self.model.has_pending_cache():
            raise RuntimeError(
                "Cannot start a training step while a forward cache is pending."
            )
        inputs, targets = self.train_sampler.next_batch()
        return self.train_batch(inputs, targets)

    def train_batch(
        self,
        inputs: NDArray[np.integer],
        targets: NDArray[np.integer],
    ) -> TrainingStepMetrics:
        """Run one update from an explicit integer input-target batch."""
        if self.completed_steps >= self.config.maximum_steps:
            raise RuntimeError(
                f"Training already reached maximum_steps={self.config.maximum_steps}."
            )
        if self.model.has_pending_cache():
            raise RuntimeError(
                "Cannot start a training step while a forward cache is pending."
            )
        self.model.train()
        self.optimizer.zero_grad()
        try:
            logits = self.model.forward(inputs)
            loss, grad_logits = softmax_cross_entropy_loss_and_gradient(
                logits,
                targets,
            )
            self.model.backward(grad_logits)
        except Exception:
            self.model.clear_cache()
            raise

        if self.config.weight_decay:
            for parameter in self.model.parameters():
                parameter.grad += self.config.weight_decay * parameter.data
        pre_clipping_norm = global_gradient_norm(self.model.parameters())
        if self.config.maximum_gradient_norm is not None:
            clip_grad_norm(
                self.model.parameters(),
                self.config.maximum_gradient_norm,
            )
        post_clipping_norm = global_gradient_norm(self.model.parameters())
        self.optimizer.step()
        self.completed_steps += 1
        return TrainingStepMetrics(
            step=self.completed_steps,
            loss=loss,
            perplexity=safe_perplexity(loss),
            pre_clipping_gradient_norm=pre_clipping_norm,
            post_clipping_gradient_norm=post_clipping_norm,
            learning_rate=self.optimizer.learning_rate,
        )

    def evaluate(self) -> dict[str, EvaluationMetrics]:
        """Evaluate fixed train and validation batches without changing continuation."""
        train_metrics = evaluate_language_model(
            self.model,
            self.train_tokens,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            batches=self.config.evaluation_batches,
            seed=self.config.seed + 2,
        )
        validation_metrics = evaluate_language_model(
            self.model,
            self.validation_tokens,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            batches=self.config.evaluation_batches,
            seed=self.config.seed + 3,
        )
        return {"train": train_metrics, "validation": validation_metrics}

    def record_evaluation(
        self,
        step_metrics: TrainingStepMetrics | None,
        evaluation: Mapping[str, EvaluationMetrics],
    ) -> dict[str, Any]:
        """Record one evaluation and update best-validation metadata."""
        if set(evaluation) != {"train", "validation"}:
            raise ValueError("evaluation must contain train and validation metrics.")
        validation = evaluation["validation"]
        if not isinstance(validation, EvaluationMetrics) or not isinstance(
            evaluation["train"],
            EvaluationMetrics,
        ):
            raise TypeError("evaluation values must be EvaluationMetrics.")
        if (
            self.best_validation_loss is None
            or validation.loss < self.best_validation_loss
        ):
            self.best_validation_loss = validation.loss
            self.best_validation_step = self.completed_steps
        record: dict[str, Any] = {
            "step": self.completed_steps,
            "train_evaluation": asdict(evaluation["train"]),
            "validation_evaluation": asdict(validation),
        }
        if step_metrics is not None:
            record["training_step"] = asdict(step_metrics)
        self.history.append(record)
        return record

    def run(self, *, until_step: int | None = None) -> list[dict[str, Any]]:
        """Train through an absolute step, evaluating and checkpointing predictably."""
        target = self.config.maximum_steps if until_step is None else until_step
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("until_step must be None or an integer.")
        if target < self.completed_steps or target > self.config.maximum_steps:
            raise ValueError(
                f"until_step must lie in [{self.completed_steps}, "
                f"{self.config.maximum_steps}]."
            )
        output_directory = Path(self.config.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        while self.completed_steps < target:
            step_metrics = self.train_step()
            should_evaluate = (
                self.completed_steps == 1
                or self.completed_steps % self.config.evaluation_interval == 0
                or self.completed_steps == target
            )
            if should_evaluate:
                previous_best = self.best_validation_loss
                self.record_evaluation(step_metrics, self.evaluate())
                if previous_best is None or self.best_validation_loss != previous_best:
                    self.save_checkpoint(
                        output_directory / "best_training_checkpoint.npz"
                    )
            elif self.completed_steps % self.config.logging_interval == 0:
                self.history.append(
                    {
                        "step": self.completed_steps,
                        "training_step": asdict(step_metrics),
                    }
                )
            if (
                self.completed_steps % self.config.checkpoint_interval == 0
                or self.completed_steps == target
            ):
                self.save_checkpoint(
                    output_directory / "latest_training_checkpoint.npz"
                )
        return [dict(record) for record in self.history]

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically persist all state required for exact training continuation."""
        if self.model.has_pending_cache():
            raise RuntimeError(
                "Cannot save a training checkpoint with a pending forward cache."
            )
        optimizer_state = self.optimizer.state_dict()
        optimizer_arrays = optimizer_state.pop("arrays")
        metadata = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "package_version": __version__,
            "checkpoint_type": "transformer_training",
            "model_type": type(self.model).__name__,
            "model_configuration": self.model.config.to_dict(),
            "model_training_mode": self.model.training,
            "optimizer_state": optimizer_state,
            "completed_steps": self.completed_steps,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_step": self.best_validation_step,
            "training_configuration": self.config.to_dict(),
            "sampler_state": self.train_sampler.state_dict(),
            "tokenizer": {
                "format_version": self.tokenizer.FORMAT_VERSION,
                "type": "character",
                "characters": list(self.tokenizer.characters),
            },
            "corpus": self.corpus_metadata,
            "history": self.history,
            "seed": self.config.seed,
            "weight_decay_convention": "coupled_l2_before_global_norm_clipping",
        }
        try:
            metadata_json = json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Training checkpoint metadata is not serializable."
            ) from error
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(metadata_json),
        }
        for name, values in self.model.state_dict().items():
            arrays[f"model::{name}"] = values
        for name, values in optimizer_arrays.items():
            arrays[f"optimizer::{name}"] = values
        return atomic_savez(path, arrays)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        train_tokens: NDArray[np.integer],
        validation_tokens: NDArray[np.integer],
        tokenizer: CharacterTokenizer,
        expected_model_config: TransformerConfig | None = None,
        expected_training_config: TransformerTrainingConfig | None = None,
    ) -> TransformerTrainer:
        """Reconstruct and validate a full training checkpoint transactionally."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as checkpoint:
                if "metadata_json" not in checkpoint.files:
                    raise ValueError("Training checkpoint is missing metadata_json.")
                try:
                    metadata = json.loads(str(checkpoint["metadata_json"]))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Training checkpoint metadata is not valid JSON."
                    ) from error
                arrays = {
                    key: np.array(checkpoint[key], copy=True)
                    for key in checkpoint.files
                    if key != "metadata_json"
                }
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Training checkpoint does not exist: {source}"
            ) from None
        if not isinstance(metadata, dict):
            raise ValueError("Training checkpoint metadata must be an object.")
        expected_metadata_keys = {
            "checkpoint_version",
            "package_version",
            "checkpoint_type",
            "model_type",
            "model_configuration",
            "model_training_mode",
            "optimizer_state",
            "completed_steps",
            "best_validation_loss",
            "best_validation_step",
            "training_configuration",
            "sampler_state",
            "tokenizer",
            "corpus",
            "history",
            "seed",
            "weight_decay_convention",
        }
        if set(metadata) != expected_metadata_keys:
            raise ValueError(
                "Training checkpoint metadata keys do not match the expected format."
            )
        if metadata.get("checkpoint_version") != cls.CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported training checkpoint version: "
                f"{metadata.get('checkpoint_version')!r}."
            )
        if metadata.get("package_version") != __version__:
            raise ValueError(
                f"Training checkpoint package version "
                f"{metadata.get('package_version')!r} does not match "
                f"{__version__!r}."
            )
        if metadata.get("checkpoint_type") != "transformer_training":
            raise ValueError("Checkpoint is not a full transformer training state.")
        if metadata.get("model_type") != TransformerLanguageModel.__name__:
            raise ValueError("Training checkpoint model type is incompatible.")
        if (
            metadata.get("weight_decay_convention")
            != "coupled_l2_before_global_norm_clipping"
        ):
            raise ValueError("Training checkpoint weight-decay convention is invalid.")

        model_config = TransformerConfig.from_dict(metadata.get("model_configuration"))
        training_config = TransformerTrainingConfig.from_dict(
            metadata.get("training_configuration")
        )
        if metadata.get("seed") != training_config.seed:
            raise ValueError("Training checkpoint seed metadata is inconsistent.")
        if expected_model_config is not None and model_config != expected_model_config:
            raise ValueError("Training checkpoint model configuration is incompatible.")
        if (
            expected_training_config is not None
            and training_config != expected_training_config
        ):
            if (
                training_config.optimizer_name
                != expected_training_config.optimizer_name
            ):
                raise ValueError(
                    "Training checkpoint optimizer selection is incompatible."
                )
            raise ValueError(
                "Training checkpoint training configuration is incompatible."
            )
        tokenizer_metadata = metadata.get("tokenizer")
        expected_tokenizer_metadata = {
            "format_version": tokenizer.FORMAT_VERSION,
            "type": "character",
            "characters": list(tokenizer.characters),
        }
        if tokenizer_metadata != expected_tokenizer_metadata:
            raise ValueError(
                "Training checkpoint tokenizer vocabulary is incompatible."
            )

        train_stream = _validated_token_stream(
            train_tokens,
            name="train_tokens",
            vocabulary_size=model_config.vocabulary_size,
        )
        validation_stream = _validated_token_stream(
            validation_tokens,
            name="validation_tokens",
            vocabulary_size=model_config.vocabulary_size,
        )
        expected_corpus = {
            "train_token_count": int(train_stream.size),
            "train_sha256": _token_stream_digest(train_stream),
            "validation_token_count": int(validation_stream.size),
            "validation_sha256": _token_stream_digest(validation_stream),
        }
        if metadata.get("corpus") != expected_corpus:
            raise ValueError("Training checkpoint corpus identity is incompatible.")

        model_state = {
            name.removeprefix("model::"): values
            for name, values in arrays.items()
            if name.startswith("model::")
        }
        optimizer_arrays = {
            name.removeprefix("optimizer::"): values
            for name, values in arrays.items()
            if name.startswith("optimizer::")
        }
        unexpected_arrays = [
            name
            for name in arrays
            if not name.startswith("model::") and not name.startswith("optimizer::")
        ]
        if unexpected_arrays:
            raise ValueError(
                f"Training checkpoint has unexpected arrays: {unexpected_arrays}."
            )

        model = TransformerLanguageModel(model_config)
        model.load_state_dict(model_state)
        trainer = cls(
            model,
            tokenizer,
            train_stream,
            validation_stream,
            training_config,
        )
        optimizer_metadata = metadata.get("optimizer_state")
        if not isinstance(optimizer_metadata, Mapping):
            raise ValueError("Training checkpoint optimizer state is malformed.")
        trainer.optimizer.load_state_dict(
            {**dict(optimizer_metadata), "arrays": optimizer_arrays}
        )
        trainer.train_sampler.load_state_dict(metadata.get("sampler_state"))

        completed_steps = metadata.get("completed_steps")
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or not 0 <= completed_steps <= training_config.maximum_steps
        ):
            raise ValueError("Training checkpoint completed_steps is invalid.")
        best_loss = metadata.get("best_validation_loss")
        best_step = metadata.get("best_validation_step")
        if best_loss is None or best_step is None:
            if best_loss is not None or best_step is not None:
                raise ValueError("Best validation metadata must be both set or null.")
        else:
            if (
                isinstance(best_loss, bool)
                or not isinstance(best_loss, (int, float))
                or not np.isfinite(best_loss)
                or best_loss < 0.0
            ):
                raise ValueError("Best validation loss is invalid.")
            if (
                isinstance(best_step, bool)
                or not isinstance(best_step, int)
                or not 0 <= best_step <= completed_steps
            ):
                raise ValueError("Best validation step is invalid.")
            best_loss = float(best_loss)
        history = metadata.get("history")
        if not isinstance(history, list) or not all(
            isinstance(record, dict) for record in history
        ):
            raise ValueError("Training checkpoint history must be a list of objects.")
        mode = metadata.get("model_training_mode")
        if not isinstance(mode, bool):
            raise ValueError("Training checkpoint model mode must be boolean.")

        trainer.completed_steps = completed_steps
        trainer.best_validation_loss = best_loss
        trainer.best_validation_step = best_step
        trainer.history = [dict(record) for record in history]
        if mode:
            trainer.model.train()
        else:
            trainer.model.eval()
        return trainer
