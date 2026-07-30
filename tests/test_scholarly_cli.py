from __future__ import annotations

import json

import pytest

from localml_scholar.scholarly.cli import run
from localml_scholar.scholarly.serialization import load_artifact


def _document(index, source):
    return next(item for item in index.documents if item.source_name == source)


@pytest.mark.parametrize(
    "command",
    [
        "analyze",
        "glossary",
        "equations",
        "methods",
        "experiments",
        "summary",
        "reproduction-checklist",
        "inspect",
    ],
)
def test_single_paper_cli_commands(
    command,
    scholarly_index,
    tmp_path,
    capsys,
) -> None:
    index_path = scholarly_index.save(tmp_path / "index.json")
    document = _document(scholarly_index, "sparse_gate_network.md")
    output = tmp_path / f"{command}.json"

    assert (
        run(
            [
                command,
                "--index",
                str(index_path),
                "--document-id",
                document.document_id,
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)
    assert load_artifact(output, index=scholarly_index).artifact_type


def test_comparison_and_gap_cli_commands(scholarly_index, tmp_path, capsys) -> None:
    index_path = scholarly_index.save(tmp_path / "index.json")
    ids = [item.document_id for item in scholarly_index.documents[:2]]
    for command in ("compare", "research-gaps"):
        arguments = [command, "--index", str(index_path), "--json"]
        for document_id in ids:
            arguments.extend(["--document-id", document_id])
        assert run(arguments) == 0
        assert json.loads(capsys.readouterr().out)


def test_cli_rejects_absent_section_role(scholarly_index, tmp_path) -> None:
    index_path = scholarly_index.save(tmp_path / "index.json")
    document = _document(scholarly_index, "dense_gate_companion.md")
    with pytest.raises(ValueError, match="section role"):
        run(
            [
                "analyze",
                "--index",
                str(index_path),
                "--document-id",
                document.document_id,
                "--section-role",
                "background",
            ]
        )


def test_cli_section_role_returns_only_matching_sections(
    scholarly_index, tmp_path, capsys
) -> None:
    index_path = scholarly_index.save(tmp_path / "index.json")
    document = _document(scholarly_index, "sparse_gate_network.md")
    output = tmp_path / "appendix.json"
    assert (
        run(
            [
                "analyze",
                "--index",
                str(index_path),
                "--document-id",
                document.document_id,
                "--section-role",
                "appendix",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["section_role"] == "appendix"
    assert all("appendix" in item["roles"] for item in payload["sections"])
    assert {item["equation_number"] for item in payload["equations"]} == {"3"}
    assert load_artifact(output, index=scholarly_index).artifact_type == (
        "paper_analysis_section"
    )
