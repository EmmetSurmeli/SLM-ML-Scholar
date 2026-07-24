from pathlib import Path

import numpy as np

from experiments.compare_tokenizers import compare_tokenizers
from experiments.inspect_bpe_tokenizer import inspect_bpe


def test_bpe_inspection_exposes_hand_checked_trace(tmp_path: Path) -> None:
    summary = inspect_bpe(output_directory=tmp_path / "inspection")

    assert summary["round_trip_exact"]
    assert [step["selected_pair"] for step in summary["merge_trace"]] == [
        [97, 110],
        [98, 256],
        [256, 97],
    ]
    assert summary["encoded_tokens"] == [257, 258, 32, 257, 100, 258]
    assert Path(summary["artifacts"]["summary"]).is_file()
    assert Path(summary["artifacts"]["tokenizer"]).is_file()


def test_controlled_comparison_runs_all_tokenizers(tmp_path: Path) -> None:
    summary = compare_tokenizers(
        seed=149,
        steps=2,
        output_directory=tmp_path / "comparison",
    )

    assert "not directly comparable" in summary["perplexity_warning"]
    assert [run["tokenizer_type"] for run in summary["runs"]] == [
        "character",
        "byte",
        "bpe",
    ]
    for run in summary["runs"]:
        assert run["round_trip_exact"]
        assert run["checkpoint_reload_logits_equal"]
        assert run["resumed_to_step"] == 2
        assert run["training_token_count"] > 0
        assert run["validation_token_count"] > 0
        assert run["training_average_bytes_per_token"] > 0.0
        assert np.isfinite(run["byte_normalized_validation"]["bits_per_byte"])
    byte_run = summary["runs"][1]
    bpe_run = summary["runs"][2]
    assert byte_run["vocabulary_size"] == 256
    assert byte_run["learned_merges"] == 0
    assert bpe_run["vocabulary_size"] > 256
    assert bpe_run["learned_merges"] == bpe_run["vocabulary_size"] - 256
    assert Path(summary["summary_path"]).is_file()
