"""Service-level tests for the localhost paper review workflow."""

from __future__ import annotations

import json
from io import BytesIO

import pytest

from localml_scholar.retrieval import PageText, RetrievalIndex
from localml_scholar.review_app.service import ReviewService

_PAPER = b"""# Tiny Optimizer Study

## Abstract
We study a local gradient balance method for small classifiers.

## Method
The model is a two-layer neural network. Training uses the Adam optimizer with
learning rate 0.001 for 50 epochs. The batch size is 32.

## Experiments
We evaluate on the Tiny Shapes dataset using accuracy and compare against
stochastic gradient descent.

## Limitations
The method was tested only on synthetic images and may not transfer to natural
images.
"""


def test_add_list_analyze_and_replace_paper(tmp_path):
    service = ReviewService(tmp_path)
    first = service.add_paper(filename="study.md", payload=_PAPER)

    assert first["title"] == "Tiny Optimizer Study"
    assert first["section_count"] == 5
    assert service.list_papers() == [first]
    assert (tmp_path / first["source_path"]).read_bytes() == _PAPER
    assert (
        RetrievalIndex.load(service.index_path).documents[0].document_id
        == (first["document_id"])
    )

    analysis = service.analyze(first["document_id"])
    assert analysis["paper"] == first
    assert analysis["analysis"]["paper"]["title"]["value"] == "Tiny Optimizer Study"
    assert analysis["summary"]["paper_id"] == analysis["analysis"]["paper"]["paper_id"]
    assert (
        analysis["checklist"]["paper_id"] == analysis["analysis"]["paper"]["paper_id"]
    )
    assert "Adam optimizer" in analysis["source"]["text"]

    replacement = service.add_paper(
        filename="study.md",
        payload=_PAPER.replace(b"50 epochs", b"75 epochs"),
    )
    assert replacement["document_id"] != first["document_id"]
    assert len(service.list_papers()) == 1
    index = RetrievalIndex.load(service.index_path)
    assert len(index.documents) == 1
    assert "75 epochs" in index.documents[0].text


def test_question_and_feedback_preserve_exact_snapshot(tmp_path):
    service = ReviewService(tmp_path)
    paper = service.add_paper(filename="study.md", payload=_PAPER)

    interaction = service.ask(
        question="What optimizer and learning rate are used?",
        document_id=paper["document_id"],
        audience_level="researcher",
    )
    assert interaction["audience_level"] == "researcher"
    assert interaction["answer"]["abstained"] is False
    assert "Adam optimizer" in interaction["answer"]["answer_text"]
    assert "[C1]" in interaction["answer"]["answer_text"]
    assert interaction["answer"]["evidence"]

    feedback = service.save_feedback(
        interaction_id=interaction["interaction_id"],
        verdict="partially_correct",
        issue_categories=["missing_method_detail"],
        notes="Include the epoch count.",
        corrected_answer="Adam, learning rate 0.001, for 50 epochs.",
        audience_level="beginner",
    )
    assert feedback["status"] == "pending_codex_review"
    assert feedback["audience_level"] == "beginner"
    assert feedback["interaction"] == interaction
    persisted = json.loads(service.feedback_path.read_text(encoding="utf-8"))
    assert persisted == [feedback]
    state = service.state()
    assert state["interaction_count"] == 1
    assert state["feedback_count"] == 1
    assert state["feedback"] == [feedback]
    assert state["storage"]["feedback"] == str(service.feedback_path)


@pytest.mark.parametrize(
    "audience_level",
    [
        "researcher",
        "undergraduate",
        "beginner",
    ],
)
def test_every_audience_level_round_trips_through_interaction_and_feedback(
    tmp_path,
    audience_level,
):
    service = ReviewService(tmp_path)
    paper = service.add_paper(filename="study.md", payload=_PAPER)
    interaction = service.ask(
        question="What optimizer is used?",
        document_id=paper["document_id"],
        audience_level=audience_level,
    )
    feedback = service.save_feedback(
        interaction_id=interaction["interaction_id"],
        verdict="correct",
        issue_categories=[],
    )

    assert interaction["audience_level"] == audience_level
    assert feedback["audience_level"] == audience_level


def test_legacy_audience_inputs_are_normalized(tmp_path):
    service = ReviewService(tmp_path)
    service.add_paper(filename="study.md", payload=_PAPER)
    interaction = service.ask(
        question="What does the paper say?",
        audience_level="high_school_beginner",
    )
    assert interaction["audience_level"] == "beginner"


def test_pdf_adapter_preserves_pages_without_requiring_parser(tmp_path, monkeypatch):
    service = ReviewService(tmp_path)

    def extracted(_payload):
        return (
            (
                PageText(1, "First page introduces the Alpha method."),
                PageText(2, "Second page states the method uses Adam optimizer."),
            ),
            "Alpha Paper",
        )

    monkeypatch.setattr(service, "_extract_pdf", extracted)
    paper = service.add_paper(filename="alpha.pdf", payload=b"%PDF-fixture")
    assert paper["title"] == "Alpha Paper"
    assert paper["page_count"] == 2

    index = RetrievalIndex.load(service.index_path)
    assert index.documents[0].media_type == "application/pdf-derived-text"
    assert [section.page_start for section in index.documents[0].sections] == [1, 2]


def test_real_pdf_text_extraction_stays_local_and_page_aware(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(
        b"BT /F1 12 Tf 40 120 Td "
        b"(Training uses Adam optimizer at learning rate 0.001.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    stream = BytesIO()
    writer.write(stream)

    service = ReviewService(tmp_path)
    paper = service.add_paper(filename="optimizer.pdf", payload=stream.getvalue())
    assert paper["page_count"] == 1
    index = RetrievalIndex.load(service.index_path)
    assert "Adam optimizer" in index.documents[0].text
    assert index.documents[0].sections[0].page_start == 1


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("paper.docx", b"text", "Supported paper files"),
        ("paper.txt", b"", "empty"),
        ("paper.txt", b"   \n", "non-whitespace"),
        ("paper.txt", b"\xff", "valid UTF-8"),
        ("../", b"text", "filename"),
    ],
)
def test_upload_validation(tmp_path, filename, payload, message):
    service = ReviewService(tmp_path)
    with pytest.raises(ValueError, match=message):
        service.add_paper(filename=filename, payload=payload)


def test_question_and_feedback_validation(tmp_path):
    service = ReviewService(tmp_path)
    with pytest.raises(ValueError, match="No papers"):
        service.ask(question="What is the method?")

    paper = service.add_paper(filename="study.md", payload=_PAPER)
    with pytest.raises(ValueError, match="Unknown document_id"):
        service.ask(question="What is the method?", document_id="missing")
    interaction = service.ask(
        question="What optimizer is used?",
        document_id=paper["document_id"],
    )
    assert interaction["audience_level"] == "undergraduate"
    with pytest.raises(ValueError, match="audience_level"):
        service.ask(
            question="What optimizer is used?",
            document_id=paper["document_id"],
            audience_level="expert",
        )
    with pytest.raises(ValueError, match="select at least one issue"):
        service.save_feedback(
            interaction_id=interaction["interaction_id"],
            verdict="incorrect",
            issue_categories=[],
        )
    with pytest.raises(ValueError, match="Unknown feedback categories"):
        service.save_feedback(
            interaction_id=interaction["interaction_id"],
            verdict="incorrect",
            issue_categories=["invented"],
        )
    with pytest.raises(ValueError, match="Unknown interaction_id"):
        service.save_feedback(
            interaction_id="missing",
            verdict="correct",
            issue_categories=[],
        )
    with pytest.raises(ValueError, match="audience_level"):
        service.save_feedback(
            interaction_id=interaction["interaction_id"],
            verdict="correct",
            issue_categories=[],
            audience_level="middle_school",
        )


def test_malformed_persisted_state_fails_clearly(tmp_path):
    service = ReviewService(tmp_path)
    service.output_directory.mkdir(parents=True)
    service.feedback_path.write_text('{"not": "a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        service.state()
