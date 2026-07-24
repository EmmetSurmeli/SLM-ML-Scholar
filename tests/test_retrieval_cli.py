import json
from pathlib import Path

from localml_scholar.retrieval.search import main


def test_cli_build_inspect_and_json_search(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.md"
    source.write_text(
        "# Attention\n\nA causal mask blocks future tokens.\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.json"

    assert (
        main(
            [
                "build",
                "--sources",
                str(source),
                "--output",
                str(index),
            ]
        )
        == 0
    )
    build_output = capsys.readouterr().out
    assert '"operation": "build"' in build_output
    assert index.is_file()

    assert main(["inspect", "--index", str(index)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["operation"] == "inspect"
    assert inspection["documents"][0]["source_name"] == "source.md"

    assert (
        main(
            [
                "search",
                "--index",
                str(index),
                "--query",
                "future mask",
                "--method",
                "bm25",
                "--top-k",
                "3",
                "--verbose",
                "--json",
            ]
        )
        == 0
    )
    search = json.loads(capsys.readouterr().out)
    assert search["operation"] == "search"
    assert search["result_count"] == 1
    assert search["results"][0]["citation"]["display"]
    assert "[[mask]]" in search["results"][0]["highlighted_text"].casefold()
    assert search["answer_generated"] is False


def test_cli_human_search_prints_exact_passage(tmp_path: Path, capsys) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Gradient descent follows a gradient.", encoding="utf-8")
    index = tmp_path / "index.json"
    main(["build", "--sources", str(source), "--output", str(index)])
    capsys.readouterr()

    main(
        [
            "search",
            "--index",
            str(index),
            "--query",
            "gradient",
            "--verbose",
        ]
    )
    output = capsys.readouterr().out

    assert "method: bm25" in output
    assert "citation:" in output
    assert "matched terms: gradient" in output
    assert "term contributions:" in output
    assert "scoring details:" in output
    assert "Gradient descent follows a gradient." in output
    assert "answer" not in output.casefold()
