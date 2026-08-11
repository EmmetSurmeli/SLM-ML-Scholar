"""Regression tests for unchanged older artifacts after package upgrades."""

from __future__ import annotations

import localml_scholar.answering.serialization as answer_serialization
from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.retrieval import RetrievalIndex, ingest_markdown
from localml_scholar.training.transformer import TransformerTrainer


def test_version_1_1_1_retrieval_index_remains_loadable(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam at learning rate 0.001.\n",
        source="paper.md",
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.1.1"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "old-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.0"
    )
    loaded = RetrievalIndex.load(path)
    assert loaded.index_sha256 == old_index.index_sha256
    assert loaded.package_version == "1.1.1"


def test_version_1_1_1_grounded_answer_remains_loadable(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam at learning rate 0.001.\n",
        source="paper.md",
    )
    index = RetrievalIndex.build([document])
    answer = GroundedAnswerPipeline(index).answer(
        "Which optimizer and learning rate are used?", method="extractive"
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.1.1")
    path = answer_serialization.save_grounded_answer(tmp_path / "answer.json", answer)
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.0")
    loaded = answer_serialization.load_grounded_answer(path, index=index)
    assert loaded == answer


def test_version_1_1_1_training_checkpoint_identity_is_supported():
    assert (3, "1.1.1") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES
    assert (3, "1.2.0") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES


def test_version_1_2_0_retrieval_index_remains_loadable(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.0"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "version-1.2.0-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.1"
    )
    loaded = RetrievalIndex.load(path)
    assert loaded.index_sha256 == old_index.index_sha256
    assert loaded.package_version == "1.2.0"


def test_version_1_2_0_grounded_answer_remains_loadable(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    index = RetrievalIndex.build([document])
    answer = GroundedAnswerPipeline(index).answer(
        "Which optimizer is used?", method="extractive"
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.0")
    path = answer_serialization.save_grounded_answer(tmp_path / "answer.json", answer)
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.1")
    assert answer_serialization.load_grounded_answer(path, index=index) == answer


def test_version_1_2_1_artifacts_remain_supported(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.1"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "version-1.2.1-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.2"
    )
    assert RetrievalIndex.load(path).index_sha256 == old_index.index_sha256
    assert (3, "1.2.1") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES


def test_version_1_2_1_grounded_answer_remains_loadable(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    index = RetrievalIndex.build([document])
    answer = GroundedAnswerPipeline(index).answer(
        "Which optimizer is used?", method="extractive"
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.1")
    path = answer_serialization.save_grounded_answer(
        tmp_path / "answer-1.2.1.json", answer
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.2")
    assert answer_serialization.load_grounded_answer(path, index=index) == answer


def test_version_1_2_3_artifacts_remain_supported(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.3"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "version-1.2.3-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.4"
    )
    assert RetrievalIndex.load(path).index_sha256 == old_index.index_sha256
    assert (3, "1.2.3") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES


def test_version_1_2_4_artifacts_remain_supported(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.4"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "version-1.2.4-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.5"
    )
    assert RetrievalIndex.load(path).index_sha256 == old_index.index_sha256
    assert (3, "1.2.4") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES

    answer = GroundedAnswerPipeline(old_index).answer(
        "Which optimizer is used?", method="extractive"
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.4")
    answer_path = answer_serialization.save_grounded_answer(
        tmp_path / "answer-1.2.4.json", answer
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.5")
    assert (
        answer_serialization.load_grounded_answer(answer_path, index=old_index)
        == answer
    )


def test_version_1_2_5_artifacts_remain_supported(tmp_path, monkeypatch):
    document = ingest_markdown(
        "# Paper\n\n## Method\nTraining uses Adam.\n", source="paper.md"
    )
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.5"
    )
    old_index = RetrievalIndex.build([document])
    path = tmp_path / "version-1.2.5-index.json"
    old_index.save(path)
    monkeypatch.setitem(
        RetrievalIndex.__init__.__kwdefaults__, "package_version", "1.2.6"
    )
    assert RetrievalIndex.load(path).index_sha256 == old_index.index_sha256
    assert (3, "1.2.5") in TransformerTrainer.LEGACY_CHECKPOINT_IDENTITIES

    answer = GroundedAnswerPipeline(old_index).answer(
        "Which optimizer is used?", method="extractive"
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.5")
    answer_path = answer_serialization.save_grounded_answer(
        tmp_path / "answer-1.2.5.json", answer
    )
    monkeypatch.setattr(answer_serialization, "__version__", "1.2.6")
    assert (
        answer_serialization.load_grounded_answer(answer_path, index=old_index)
        == answer
    )
