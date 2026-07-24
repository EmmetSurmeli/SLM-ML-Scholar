from pathlib import Path

from experiments.compare_single_and_multi_head import (
    compare_single_and_multi_head,
)
from experiments.inspect_multi_head_attention import (
    inspect_multi_head_attention,
)


def test_multi_head_inspection_experiment(tmp_path: Path) -> None:
    summary = inspect_multi_head_attention(
        seed=109,
        output_directory=tmp_path / "inspection",
    )

    assert summary["future_probabilities_exactly_zero"]
    assert summary["earlier_outputs_unchanged_after_future_change"]
    assert summary["one_head_maximum_absolute_difference"] == 0.0
    assert summary["shapes"]["query_per_head"] == [1, 2, 4, 2]
    assert Path(summary["summary_path"]).is_file()


def test_single_and_multi_head_comparison_experiment(tmp_path: Path) -> None:
    summary = compare_single_and_multi_head(
        seed=113,
        steps=3,
        output_directory=tmp_path / "comparison",
    )

    one_head, two_heads = summary["runs"]
    assert one_head["number_of_heads"] == 1
    assert two_heads["number_of_heads"] == 2
    assert two_heads["parameter_count"] > one_head["parameter_count"]
    assert one_head["checkpoint_reload_logits_equal"]
    assert two_heads["checkpoint_reload_logits_equal"]
    assert Path(summary["summary_path"]).is_file()
