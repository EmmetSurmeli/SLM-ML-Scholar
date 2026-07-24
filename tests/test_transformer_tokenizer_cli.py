from pathlib import Path

import pytest

from experiments.train_transformer_lm import parse_args, run_training
from localml_scholar.tokenizer import (
    BPETrainingConfig,
    BytePairTokenizer,
    load_tokenizer,
)


def test_new_bpe_cli_fits_saves_and_reports_tokenizer(tmp_path: Path) -> None:
    summary = run_training(
        parse_args(
            [
                "--tokenizer",
                "bpe",
                "--bpe-vocabulary-size",
                "260",
                "--bpe-minimum-frequency",
                "2",
                "--steps",
                "1",
                "--evaluation-interval",
                "1",
                "--evaluation-batches",
                "1",
                "--checkpoint-interval",
                "1",
                "--generation-length",
                "1",
                "--output",
                str(tmp_path / "run"),
            ]
        )
    )

    tokenizer = load_tokenizer(summary["artifacts"]["tokenizer"])
    assert isinstance(tokenizer, BytePairTokenizer)
    assert summary["tokenizer"]["type"] == "bpe"
    assert summary["tokenizer"]["vocabulary_size"] == 260
    assert summary["tokenizer"]["state_sha256"] == tokenizer.state_hash()


def test_non_bpe_new_run_rejects_bpe_fitting_options(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--tokenizer",
            "byte",
            "--bpe-vocabulary-size",
            "260",
            "--output",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="require --tokenizer bpe"):
        run_training(args)


def test_loaded_tokenizer_rejects_bpe_refitting_options(tmp_path: Path) -> None:
    tokenizer_path = BytePairTokenizer.train(
        "banana bandana",
        BPETrainingConfig(target_vocabulary_size=257),
    ).save(tmp_path / "tokenizer.json")
    args = parse_args(
        [
            "--tokenizer-load",
            str(tokenizer_path),
            "--bpe-vocabulary-size",
            "260",
            "--output",
            str(tmp_path / "run"),
        ]
    )

    with pytest.raises(ValueError, match="loaded"):
        run_training(args)
