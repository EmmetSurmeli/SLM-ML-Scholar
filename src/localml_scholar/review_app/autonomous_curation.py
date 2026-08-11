"""Resumable application orchestration for fully automated corpus curation."""

from __future__ import annotations

import os
import re
import uuid
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from localml_scholar._version import __version__
from localml_scholar.answering import GroundedAnswerPipeline
from localml_scholar.retrieval import SearchFilters, section_topics_compatible
from localml_scholar.review_app.storage import (
    atomic_write_json,
    load_json_list,
    load_json_object,
)
from localml_scholar.training_data import (
    AutonomousCurationConfig,
    AutoReviewPolicy,
    CurationSuspended,
    GroundedInstructionExample,
    QuestionCandidate,
    ReviewProvenance,
    StructuredGroundedTarget,
    autonomous_quality_report,
    autonomous_training_exclusion,
    balanced_paper_splits,
    build_dataset,
    generate_paper_questions,
    propose_correction,
    review_interaction_second_pass,
    save_dataset,
    select_diagnostic_candidates,
    should_stop_for_reliability,
)
from localml_scholar.training_data.preflight import (
    DeterministicPreflightCache,
    infer_scholarly_headings,
    paper_ingestion_health,
    topic_signals,
)
from localml_scholar.training_data.provenance import content_sha256

if TYPE_CHECKING:
    from localml_scholar.review_app.service import ReviewService


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _unique_candidate_states(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve candidate order while removing repeated stable question IDs."""
    seen: set[str] = set()
    unique = []
    for item in items:
        identity = item.get("question_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Candidate question_id must contain text.")
        if identity not in seen:
            unique.append(item)
            seen.add(identity)
    return unique


_STABLE_AUTONOMOUS_ANSWERABLE_TYPES = frozenset(
    {
        "architecture",
        "complexity",
        "method",
        "motivation",
        "reproduction",
        "section_comprehension",
    }
)


class AutonomousCorpusCurator:
    """Coordinate local answering and genuine Codex review without human gates."""

    def __init__(self, service: ReviewService) -> None:
        self.service = service
        self.state_path = service.autonomous_runs_path
        self.output_directory = service.autonomous_output_directory

    def _load_runs(self) -> list[dict[str, Any]]:
        runs = load_json_list(self.state_path)
        for run in runs:
            if not isinstance(run.get("run_id"), str) or not isinstance(
                run.get("records"), list
            ):
                raise ValueError("Autonomous curation state contains a malformed run.")
        return runs

    def _save_runs(self, runs: list[dict[str, Any]]) -> None:
        atomic_write_json(self.state_path, runs)

    def _persist(self, run: dict[str, Any]) -> None:
        run["updated_at"] = _timestamp()
        with self.service._lock:
            runs = self._load_runs()
            for position, current in enumerate(runs):
                if current["run_id"] == run["run_id"]:
                    runs[position] = run
                    self._save_runs(runs)
                    return
        raise RuntimeError(f"Autonomous run disappeared: {run['run_id']}")

    def _paper_preflight(self, document) -> dict[str, Any]:
        """Load or compute deterministic paper metadata by exact source hash."""
        source_hash = content_sha256(
            {
                "document_id": document.document_id,
                "content_sha256": document.content_sha256,
                "parser_identifier": document.parser_identifier,
                "sections": [section.to_dict() for section in document.sections],
            }
        )
        with self.service._lock:
            cache = DeterministicPreflightCache(
                load_json_object(self.service.preflight_cache_path)
            )
            cached = cache.get(source_hash)
            if cached is not None:
                return {**cached, "cache_hit": True, "source_hash": source_hash}
            inferred_titles = tuple(
                item.title for item in infer_scholarly_headings(document.text)
            )
            section_titles = (
                tuple(
                    section.heading for section in document.sections if section.heading
                )
                or inferred_titles
            )
            value = {
                "health": paper_ingestion_health(document).to_dict(),
                "section_titles": list(section_titles),
                "topic_signals": topic_signals(document.text, section_titles),
            }
            cache.put(source_hash, value)
            atomic_write_json(self.service.preflight_cache_path, cache.to_dict())
        return {**value, "cache_hit": False, "source_hash": source_hash}

    def list_runs(self) -> list[dict[str, Any]]:
        """Return newest runs first without exposing mutable internal objects."""
        with self.service._lock:
            return list(reversed(self._load_runs()))

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Load exactly one autonomous run."""
        matches = [item for item in self._load_runs() if item["run_id"] == run_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown autonomous curation run: {run_id}")
        return matches[0]

    def invalidate_for_readiness(self, run_id: str, reason: str) -> dict[str, Any]:
        """Freeze a historical diagnostic without rewriting its work artifacts."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must contain text.")
        run = self.get_run(run_id)
        diagnostic = run.get("diagnostic")
        if not isinstance(diagnostic, dict) or not diagnostic.get("controlled"):
            raise ValueError("Only a controlled diagnostic can be invalidated.")
        run["diagnostic"] = {
            **diagnostic,
            "valid_for_readiness": False,
            "invalidation_reason": reason.strip(),
        }
        run["historical_state_marker"] = reason.strip()
        run["status"] = "suspended"
        run["stage"] = "readiness_invalidated"
        self._persist(run)
        return run

    def start(
        self,
        *,
        paper_ids: tuple[str, ...] | None = None,
        config: AutonomousCurationConfig | None = None,
    ) -> dict[str, Any]:
        """Create, persist, and execute one conservative autonomous run."""
        run = self.create(paper_ids=paper_ids, config=config)
        return self.resume(run["run_id"])

    def create(
        self,
        *,
        paper_ids: tuple[str, ...] | None = None,
        config: AutonomousCurationConfig | None = None,
    ) -> dict[str, Any]:
        """Persist a new run without blocking while its Codex passes execute."""
        config = AutonomousCurationConfig() if config is None else config
        papers = self.service.list_papers()
        known = {item["document_id"]: item for item in papers}
        index = self.service._load_index()
        documents = {item.document_id: item for item in index.documents}
        selected = tuple(sorted(known)) if paper_ids is None else paper_ids
        if not selected:
            raise ValueError("At least one indexed paper is required for curation.")
        if not isinstance(selected, tuple) or not all(
            isinstance(item, str) and item.strip() for item in selected
        ):
            raise TypeError("paper_ids must be a tuple of non-empty strings.")
        unknown = set(selected) - set(known)
        if unknown:
            raise ValueError(f"Unknown paper IDs: {sorted(unknown)}.")
        splits = balanced_paper_splits(
            selected,
            seed=config.seed,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
        )
        source_hashes = {}
        analysis_hashes = {}
        candidates = {}
        ingestion_health = {}
        templates_suppressed = 0
        preflight_cache_hits = 0
        for paper_id in selected:
            document = documents[paper_id]
            source_hashes[paper_id] = content_sha256(document.to_dict())
            preflight = self._paper_preflight(document)
            preflight_cache_hits += int(preflight["cache_hit"])
            analysis_hashes[paper_id] = content_sha256(
                {
                    "source_hash": source_hashes[paper_id],
                    "preflight_source_hash": preflight["source_hash"],
                    "section_titles": preflight["section_titles"],
                    "topic_signals": preflight["topic_signals"],
                    "derivation": "deterministic_preflight_v1",
                }
            )
            health_state = preflight["health"]
            health = paper_ingestion_health(document)
            if health.to_dict() != health_state:
                raise RuntimeError("Cached paper-ingestion health is inconsistent.")
            ingestion_health[paper_id] = health.to_dict()
            section_titles = tuple(preflight["section_titles"])
            generated = (
                generate_paper_questions(
                    paper_id,
                    document.title or document.source_name,
                    count=80,
                    section_titles=section_titles,
                    paper_text=document.text,
                )
                if health.healthy_for_question_generation
                else ()
            )
            templates_suppressed += max(0, 80 - len(generated))
            eligible = [
                item.to_dict()
                for item in generated
                if self._candidate_enabled(item.question_type, config)
            ]
            candidates[paper_id] = eligible[: config.questions_per_paper]
            section_candidates = []
            for heading in (
                section_titles[:10] if health.healthy_for_question_generation else ()
            ):
                section_candidates.append(
                    QuestionCandidate.create(
                        paper_ids=(paper_id,),
                        question=(
                            f"In the section '{heading}', what does the paper "
                            "state, and which local evidence supports it?"
                        ),
                        question_type="section_comprehension",
                        expected_sections=(heading,),
                        metadata={
                            "source": "content_adaptive_section_generation",
                            "candidate_only": True,
                            "analysis_hash": analysis_hashes[paper_id],
                        },
                    ).to_dict()
                )
            section_candidates = _unique_candidate_states(section_candidates)
            if section_candidates:
                keep = max(0, config.questions_per_paper - len(section_candidates))
                candidates[paper_id] = (
                    section_candidates + candidates[paper_id][:keep]
                )[: config.questions_per_paper]
            if config.include_multi_turn and candidates[paper_id]:
                parent = candidates[paper_id][0]
                follow_up = QuestionCandidate.create(
                    paper_ids=(paper_id,),
                    question=(
                        "I understand the basic idea. Now explain it more "
                        "mathematically, while keeping every claim grounded in "
                        "this paper."
                    ),
                    question_type="multi_turn_follow_up",
                    parent_question_id=parent["question_id"],
                    metadata={
                        "source": "autonomous_multi_turn_generation",
                        "candidate_only": True,
                    },
                )
                candidates[paper_id][-1] = follow_up.to_dict()
        if config.include_cross_paper and len(selected) > 1:
            for left, right in zip(selected, selected[1:], strict=False):
                cross = QuestionCandidate.create(
                    paper_ids=(left, right),
                    question=(
                        "Compare the central methods in these two papers, including "
                        "their assumptions and experimental evidence."
                    ),
                    question_type="cross_paper_comparison",
                    metadata={
                        "source": "autonomous_cross_paper_generation",
                        "candidate_only": True,
                        "requires_all_local_sources": True,
                    },
                )
                candidates[left].append(cross.to_dict())
        run = {
            "run_id": _identifier("curation"),
            "format_version": "1.0",
            "package_version": __version__,
            "status": "running",
            "stage": "deterministic_preflight_complete",
            "stage_id": _identifier("stage"),
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "config": config.to_dict(),
            "paper_ids": list(selected),
            "paper_splits": splits,
            "source_hashes": source_hashes,
            "analysis_hashes": analysis_hashes,
            "ingestion_health": ingestion_health,
            "preflight_metrics": {
                "healthy_papers": sum(
                    item["healthy_for_question_generation"]
                    for item in ingestion_health.values()
                ),
                "unhealthy_papers": sum(
                    not item["healthy_for_question_generation"]
                    for item in ingestion_health.values()
                ),
                "question_templates_suppressed": templates_suppressed,
                "preflight_cache_hits": preflight_cache_hits,
                "preflight_cache_misses": len(selected) - preflight_cache_hits,
            },
            "candidates": candidates,
            "cursor": {"paper_index": 0, "question_index": 0},
            "records": [],
            "evaluation_questions": [],
            "errors": [],
            "reviewer": {
                "available": self.service.codex_provider.available(),
                "identity": list(self.service.codex_provider.identity),
                "genuine_codex_required": True,
            },
            "report": None,
            "dataset_path": None,
            "manifest_path": None,
        }
        with self.service._lock:
            runs = self._load_runs()
            runs.append(run)
            self._save_runs(runs)
        return run

    @staticmethod
    def _candidate_enabled(
        question_type: str, config: AutonomousCurationConfig
    ) -> bool:
        if question_type in {"derivation", "equation"}:
            return config.include_derivations
        if question_type in {"false_premise", "insufficient_evidence"}:
            return config.include_abstentions
        if question_type == "comparison":
            return config.include_cross_paper
        return True

    def _retrieval_preflight_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Keep candidates whose expected answerability matches local retrieval."""
        index = self.service._load_index()
        pipeline = GroundedAnswerPipeline(index)
        accepted: list[dict[str, Any]] = []
        rejected = 0
        for candidate in candidates:
            paper_ids = candidate.get("paper_ids", [])
            if not isinstance(paper_ids, list) or len(paper_ids) != 1:
                rejected += 1
                continue
            answer = pipeline.answer(
                str(candidate.get("question", "")),
                method="extractive",
                top_k=8,
                filters=SearchFilters(document_id=str(paper_ids[0])),
                expected_sections=tuple(candidate.get("expected_sections", [])),
            )
            expected = candidate.get("expected_answerability", "answerable")
            expected_sections = tuple(candidate.get("expected_sections", []))
            evidence_by_label = {item.label: item for item in answer.evidence}
            claim_sections_aligned = all(
                not claim.substantive
                or bool(claim.citation_labels)
                and all(
                    label in evidence_by_label
                    and (
                        not expected_sections
                        or section_topics_compatible(
                            expected_sections,
                            evidence_by_label[label].heading_path,
                        )
                    )
                    for label in claim.citation_labels
                )
                for claim in answer.claims
            )
            answer_text = answer.answer_text.casefold()
            question_type = str(candidate.get("question_type", ""))
            numeric_value_required = question_type == "reproduction" and bool(
                re.search(
                    r"\b(?:batch size|learning rate|dropout|warmup|"
                    r"how many|how long)\b",
                    str(candidate.get("question", "")).casefold(),
                )
            )
            answer_without_citations = re.sub(r"\[c\d+\]", "", answer_text)
            numeric_value_present = bool(re.search(r"\d", answer_without_citations))
            complexity_characterized = question_type != "complexity" or bool(
                re.search(
                    r"\b(?:quadratic|linear|constant|logarithmic|exponential|"
                    r"asymptotic|flops)\b|\bo\s*\(",
                    answer_text,
                )
            )
            expectation_met = (
                answer.sufficiency.sufficient
                and claim_sections_aligned
                and (not numeric_value_required or numeric_value_present)
                and complexity_characterized
                if expected == "answerable"
                else answer.abstained
                if expected in {"abstain", "external_required"}
                else True
            )
            if not expectation_met:
                rejected += 1
                continue
            state = dict(candidate)
            state["metadata"] = {
                **dict(state.get("metadata", {})),
                "retrieval_preflight_passed": True,
                "retrieval_preflight_sufficient": answer.sufficiency.sufficient,
                "retrieval_preflight_reasons": list(answer.sufficiency.reasons),
                "retrieval_preflight_evidence_ids": [
                    item.evidence_id for item in answer.evidence
                ],
            }
            accepted.append(state)
        return accepted, rejected

    def _verify_sources(self, run: dict[str, Any]) -> None:
        index = self.service._load_index()
        documents = {item.document_id: item for item in index.documents}
        for paper_id, expected in run["source_hashes"].items():
            document = documents.get(paper_id)
            if document is None:
                raise CurationSuspended(
                    f"Indexed source disappeared after run creation: {paper_id}."
                )
            actual = content_sha256(document.to_dict())
            if actual != expected:
                raise CurationSuspended(
                    f"Indexed source changed after run creation: {paper_id}."
                )

    @staticmethod
    def _as_review_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence = []
        for position, raw in enumerate(items, start=1):
            item = dict(raw)
            item["label"] = f"C{position}"
            item["evidence_id"] = str(
                item.get("evidence_id", item.get("chunk_id", f"evidence-{position}"))
            )
            evidence.append(item)
        return evidence

    def _deterministic_review(
        self,
        interaction: dict[str, Any],
        candidate_state: dict[str, Any],
    ) -> dict[str, Any]:
        candidate = QuestionCandidate.from_dict(
            {
                key: value
                for key, value in candidate_state.items()
                if key in QuestionCandidate.__dataclass_fields__
            }
        )
        return review_interaction_second_pass(
            interaction,
            candidate,
            policy=AutoReviewPolicy(
                approval_threshold=0.95,
                calibration_state="calibration_required",
            ),
        ).to_dict()

    def _curate_candidate(
        self,
        candidate_state: dict[str, Any],
        config: AutonomousCurationConfig,
    ) -> dict[str, Any]:
        from localml_scholar.training_data import curate_interaction

        paper_ids = tuple(candidate_state["paper_ids"])
        interaction = self.service.ask(
            question=candidate_state["question"],
            document_ids=paper_ids,
            audience_level=candidate_state.get("canonical_audience") or "undergraduate",
            expected_sections=tuple(candidate_state.get("expected_sections", [])),
        )
        interaction["question_id"] = candidate_state["question_id"]
        interaction["question_type"] = candidate_state["question_type"]
        expected_sections = tuple(candidate_state.get("expected_sections", []))
        evidence = interaction.get("answer", {}).get("evidence", [])
        observed_sections = tuple(
            dict.fromkeys(
                str(heading)
                for item in evidence
                if isinstance(item, dict)
                for heading in item.get("heading_path", [])
                if isinstance(heading, str) and heading.strip()
            )
        )
        if (
            expected_sections
            and evidence
            and not section_topics_compatible(expected_sections, observed_sections)
        ):
            interaction["answer"]["abstained"] = True
            interaction["answer"]["abstention_reason"] = (
                "evidence_section_topic_mismatch"
            )
        if candidate_state.get("conversation_context"):
            interaction["conversation_turns"] = candidate_state["conversation_context"]
        deterministic = self._deterministic_review(interaction, candidate_state)

        def retrieve(query: str, selected: tuple[str, ...], attempt: int):
            kind = candidate_state["question_type"]
            expansion = (
                " equation formula objective derivation symbols"
                if kind in {"derivation", "equation"}
                else f" {kind}"
            )
            expanded = f"{query}{expansion}"
            expected = candidate_state.get("expected_sections", [])
            heading = (str(expected[0]),) if attempt == 1 and expected else None
            results = self.service.search_evidence(
                query=expanded,
                paper_ids=selected,
                top_k=10,
                method="bm25",
                heading_path_prefix=heading,
            )
            if not results and heading is not None:
                results = self.service.search_evidence(
                    query=expanded,
                    paper_ids=selected,
                    top_k=10,
                    method="bm25",
                )
            return self._as_review_evidence(results)

        def revalidate(
            answer: dict[str, Any], _candidate: dict[str, Any]
        ) -> dict[str, Any]:
            working = {**interaction, "answer": answer}
            return self._deterministic_review(working, candidate_state)

        return curate_interaction(
            interaction,
            candidate_state,
            deterministic,
            provider=self.service.codex_provider,
            config=config,
            retrieve_evidence=retrieve,
            revalidate=revalidate,
        )

    @staticmethod
    def _duplicate_identity(record: dict[str, Any]) -> str:
        answer = record.get("answer", {})
        normalized = re.sub(
            r"\s+",
            " ",
            f"{record.get('question', '')} {answer.get('answer_text', '')}".casefold(),
        ).strip()
        return content_sha256(normalized)

    @staticmethod
    def _record_preflight_metrics(run: dict[str, Any], record: dict[str, Any]) -> None:
        """Accumulate call-saving and deterministic-failure diagnostics."""
        metrics = run.setdefault("preflight_metrics", {})
        calls = int(record.get("codex_call_count", 0))
        metrics["candidates_sent_to_codex"] = int(
            metrics.get("candidates_sent_to_codex", 0)
        ) + int(calls > 0)
        metrics["codex_calls"] = int(metrics.get("codex_calls", 0)) + calls
        rejected = bool(record.get("rejected_before_codex"))
        metrics["deterministic_preflight_rejections"] = int(
            metrics.get("deterministic_preflight_rejections", 0)
        ) + int(rejected)
        metrics["codex_calls_saved"] = int(metrics.get("codex_calls_saved", 0)) + (
            4 if rejected else 0
        )
        repair = record.get("answer", {}).get("deterministic_claim_repair", {})
        metrics["deterministic_repair_successes"] = int(
            metrics.get("deterministic_repair_successes", 0)
        ) + int(isinstance(repair, dict) and repair.get("outcome") == "fixed")
        if str(record.get("status", "")).endswith("_failed"):
            metrics["candidate_construction_failures"] = (
                int(metrics.get("candidate_construction_failures", 0)) + 1
            )

    def _apply_balance_policy(
        self, run: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        if record["status"] != "codex_curated":
            return record
        exclusion = autonomous_training_exclusion(
            str(record.get("question_type", "unknown"))
        )
        if exclusion is not None:
            record["status"] = "split_excluded"
            record["trust_class"] = None
            record["terminal_reasons"] = [exclusion]
            record["retained_for_evaluation"] = True
            return record
        identities = {
            item.get("duplicate_identity")
            for item in run["records"]
            if item.get("status") == "codex_curated"
        }
        identity = self._duplicate_identity(record)
        record["duplicate_identity"] = identity
        record["duplicate_cluster"] = f"exact_{identity[:20]}"
        current_terms = set(
            re.findall(
                r"[a-z0-9]+",
                (
                    f"{record.get('question', '')} "
                    f"{record.get('answer', {}).get('answer_text', '')}"
                ).casefold(),
            )
        )
        record["duplicate_terms"] = sorted(current_terms)
        near_duplicate = False
        for item in run["records"]:
            if item.get("status") != "codex_curated":
                continue
            other_terms = set(item.get("duplicate_terms", []))
            union = current_terms | other_terms
            if union and len(current_terms & other_terms) / len(union) >= 0.90:
                near_duplicate = True
                record["duplicate_cluster"] = item.get("duplicate_cluster")
                break
        if identity in identities or near_duplicate:
            record["status"] = "duplicate"
            record["trust_class"] = None
            record["terminal_reasons"] = [
                "near_duplicate" if near_duplicate else "exact_normalized_duplicate"
            ]
            return record
        cap = int(run["config"]["per_question_type_cap"])
        same_type = sum(
            item.get("status") == "codex_curated"
            and item.get("paper_ids") == record.get("paper_ids")
            and item.get("question_type") == record.get("question_type")
            for item in run["records"]
        )
        if same_type >= cap:
            record["status"] = "split_excluded"
            record["trust_class"] = None
            record["terminal_reasons"] = ["per_paper_question_type_cap"]
        return record

    def _materialize(self, record: dict[str, Any]) -> str:
        interaction = {
            "interaction_id": record["interaction_id"],
            "paper_ids": record["paper_ids"],
            "question": record["question"],
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "instruction_profile": record["instruction_profile"],
            "conversation_turns": record["conversation_context"],
            "answer": record["answer"],
        }
        candidate = propose_correction(
            interaction,
            review_label="correct",
            corrected_answer=record["answer"]["answer_text"],
            prohibited_claims=tuple(
                record.get("structured_target", {}).get("prohibited_claims", [])
            ),
            notes="Accepted by the autonomous Codex multi-pass curation pipeline.",
        )
        target_state = record.get("structured_target", {})
        if isinstance(target_state, dict) and set(target_state) & {
            "facts",
            "equations",
            "derivation_steps",
            "assumptions",
            "qualifications",
            "limitations",
            "unresolved_items",
        }:
            candidate = replace(
                candidate,
                target=StructuredGroundedTarget.from_dict(target_state),
            )
        last_pass = record["codex_review_passes"][-1]
        provenance = ReviewProvenance(
            producer_system=record["answer_producer"],
            producer_version=record["package_version"],
            reviewer_system=f"{record['final_adjudicator']}:final_adjudicator",
            reviewer_version=last_pass["reviewer_version"],
            correction_system=f"{record['final_adjudicator']}:answerer",
            source_hashes=tuple(record["source_hashes"]),
            answer_hash=content_sha256(record["answer"]["answer_text"]),
            parent_example_ids=(),
            independent_validators=(),
        )
        metadata = {
            **candidate.metadata,
            "human_approval_required": False,
            "human_approved": False,
            "trust_class": "codex_curated",
            "codex_curated": True,
            "autonomous_curation_record_id": record["curation_record_id"],
            "question_origin": record["question_origin"],
            "answer_producer": record["answer_producer"],
            "codex_review_passes": record["codex_review_passes"],
            "repair_history": record["repair_history"],
            "final_adjudicator": record["final_adjudicator"],
            "final_adjudicator_confidence": record["final_adjudicator_confidence"],
            "source_hashes": record["source_hashes"],
            "lineage_ids": record["lineage_ids"],
            "acceptance_confidence": record["final_adjudicator_confidence"],
            "package_version": record["package_version"],
            "curation_created_at": record["created_at"],
            "duplicate_cluster": record.get("duplicate_cluster"),
            "review_provenance": provenance.to_dict(),
            "approval_provenance": provenance.to_dict(),
            "review_passes_are_separate_not_independent": True,
            "supported_claim_graph": record.get("supported_claim_graph", {}),
            "claim_alignment_metrics": record.get("claim_alignment_metrics", {}),
            "citation_first_construction": True,
            "test_only": False,
        }
        accepted = replace(
            candidate,
            review_status="codex_curated",
            split=record["split"],
            metadata=metadata,
        )
        with self.service._lock:
            corrections = self.service._load_corrections()
            by_id = {item.example_id: item for item in corrections}
            by_id[accepted.example_id] = accepted
            self.service._save_corrections(list(by_id.values()))
        return accepted.example_id

    def _stop_reason(self, run: dict[str, Any]) -> str | None:
        records = run["records"]
        for item in records:
            if item.get("status") != "codex_curated":
                continue
            splits = {
                run["paper_splits"].get(paper_id)
                for paper_id in item.get("paper_ids", [])
            }
            if "test" in splits or len(splits) != 1:
                return "Paper-level split leakage was detected."
        if len(records) < 10:
            return None
        reliability_stop = should_stop_for_reliability(
            records,
            minimum_records=(
                int(run.get("diagnostic", {}).get("count", 10))
                if run.get("diagnostic", {}).get("controlled")
                else 10
            ),
            maximum_hard_disagreement_rate=float(
                run["config"].get("maximum_hard_disagreement_rate", 0.15)
            ),
            maximum_citation_structural_failure_rate=float(
                run["config"].get("maximum_citation_structural_failure_rate", 0.05)
            ),
            maximum_unresolved_support_failure_rate=float(
                run["config"].get("maximum_unresolved_support_failure_rate", 0.05)
            ),
            maximum_malformed_output_rate=float(
                run["config"].get("maximum_malformed_output_rate", 0.02)
            ),
        )
        if reliability_stop is not None:
            return reliability_stop
        reviewed = [
            item
            for item in records
            if isinstance(item.get("final_adjudicator_confidence"), (int, float))
        ]
        if reviewed:
            confidence = sum(
                float(item["final_adjudicator_confidence"]) for item in reviewed
            ) / len(reviewed)
            if confidence < float(run["config"]["acceptance_threshold"]) * 0.75:
                return (
                    "Average adjudicator confidence collapsed below the safety floor."
                )
        duplicate_rate = sum(
            item.get("status") == "duplicate" for item in records
        ) / len(records)
        if duplicate_rate > 0.50:
            return "Excessive duplicate generation triggered a safety stop."
        return None

    def create_diagnostic(self, *, count: int = 50, seed: int = 42) -> dict[str, Any]:
        """Create a fixed, type-stratified controlled diagnostic run."""
        config = AutonomousCurationConfig(
            seed=seed,
            include_multi_turn=False,
            include_derivations=False,
            include_cross_paper=False,
        )
        run = self.create(config=config)
        eligible = [
            candidate
            for paper_id, candidates in run["candidates"].items()
            if run["paper_splits"][paper_id] != "test"
            for candidate in candidates
        ]
        answerable = [
            item
            for item in eligible
            if item.get("expected_answerability", "answerable") == "answerable"
            and item.get("question_type") in _STABLE_AUTONOMOUS_ANSWERABLE_TYPES
        ]
        abstentions = [
            item
            for item in eligible
            if item.get("expected_answerability") in {"abstain", "external_required"}
        ]
        answerable, answerable_rejected = self._retrieval_preflight_candidates(
            answerable
        )
        abstentions, abstention_rejected = self._retrieval_preflight_candidates(
            abstentions
        )
        abstention_count = min(count // 5, len(abstentions))
        answerable_count = count - abstention_count
        selected = select_diagnostic_candidates(
            answerable, count=answerable_count, seed=seed
        )
        if abstention_count:
            selected.extend(
                select_diagnostic_candidates(
                    abstentions, count=abstention_count, seed=seed + 1
                )
            )
        if len(selected) != count:
            raise CurationSuspended(
                "Not enough preflight-eligible candidates for the diagnostic."
            )
        selected_by_paper = {paper_id: [] for paper_id in run["candidates"]}
        for item in selected:
            paper_ids = item.get("paper_ids", [])
            owner = str(paper_ids[0]) if paper_ids else ""
            if owner not in selected_by_paper:
                raise CurationSuspended(
                    "A selected diagnostic candidate has no owning corpus paper."
                )
            selected_by_paper[owner].append(item)
        run["candidates"] = selected_by_paper
        run["diagnostic"] = {
            "count": count,
            "seed": seed,
            "controlled": True,
            "full_run_allowed": False,
            "valid_for_readiness": True,
            "answerable_count": answerable_count,
            "abstention_count": abstention_count,
        }
        run["preflight_metrics"]["diagnostic_retrieval_preflight_rejections"] = (
            answerable_rejected + abstention_rejected
        )
        self._persist(run)
        return run

    def create_pilot(self, *, count: int = 10, seed: int = 42) -> dict[str, Any]:
        """Create a fresh answerability-separated pilot from healthy papers."""
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 5 <= count <= 20
        ):
            raise ValueError("Pilot count must be an integer in [5, 20].")
        config = AutonomousCurationConfig(
            seed=seed,
            questions_per_paper=40,
            maximum_examples_per_paper=2,
            include_multi_turn=False,
            include_derivations=False,
            include_cross_paper=False,
            include_abstentions=True,
        )
        run = self.create(config=config)

        def fail_preflight(message: str) -> None:
            run["status"] = "suspended"
            run["stage"] = "deterministic_preflight_failed"
            run["errors"].append(message)
            run["diagnostic"] = {
                "kind": "deterministic_preflight_pilot",
                "count": count,
                "seed": seed,
                "controlled": True,
                "full_run_allowed": False,
                "valid_for_readiness": False,
                "invalidation_reason": message,
            }
            self._persist(run)
            raise CurationSuspended(message)

        eligible = [
            candidate
            for paper_id, candidates in run["candidates"].items()
            if run["paper_splits"][paper_id] != "test"
            for candidate in candidates
        ]
        answerable = [
            item
            for item in eligible
            if item.get("expected_answerability", "answerable") == "answerable"
            and item.get("question_type") in _STABLE_AUTONOMOUS_ANSWERABLE_TYPES
        ]
        abstentions = [
            item
            for item in eligible
            if item.get("expected_answerability") in {"abstain", "external_required"}
        ]
        answerable, answerable_rejected = self._retrieval_preflight_candidates(
            answerable
        )
        abstentions, abstention_rejected = self._retrieval_preflight_candidates(
            abstentions
        )
        abstention_count = min(2, len(abstentions), max(1, count // 5))
        answerable_count = count - abstention_count
        if len(answerable) < answerable_count:
            fail_preflight(
                "Not enough preflight-eligible answerable candidates for the pilot."
            )
        selected = select_diagnostic_candidates(
            answerable,
            count=answerable_count,
            seed=seed,
            maximum_per_paper=1,
        )
        if len(selected) != answerable_count:
            fail_preflight(
                "Not enough paper-balanced answerable candidates for the pilot."
            )
        if abstention_count:
            selected_papers = {
                str(item["paper_ids"][0]) for item in selected if item.get("paper_ids")
            }
            balanced_abstentions = [
                item
                for item in abstentions
                if str(item["paper_ids"][0]) not in selected_papers
            ]
            if len(balanced_abstentions) < abstention_count:
                fail_preflight(
                    "Not enough paper-balanced abstention candidates for the pilot."
                )
            selected.extend(
                select_diagnostic_candidates(
                    balanced_abstentions,
                    count=abstention_count,
                    seed=seed + 1,
                    maximum_per_paper=1,
                )
            )
        if len(selected) != count:
            fail_preflight(
                "Not enough paper-balanced abstention candidates for the pilot."
            )
        selected_by_paper = {paper_id: [] for paper_id in run["candidates"]}
        for item in selected:
            selected_by_paper[str(item["paper_ids"][0])].append(item)
        if any(len(items) > 2 for items in selected_by_paper.values()):
            fail_preflight("Pilot selection exceeded two questions per paper.")
        run["candidates"] = selected_by_paper
        run["diagnostic"] = {
            "kind": "deterministic_preflight_pilot",
            "count": count,
            "seed": seed,
            "controlled": True,
            "full_run_allowed": False,
            "valid_for_readiness": True,
            "answerable_count": answerable_count,
            "abstention_count": abstention_count,
        }
        run["preflight_metrics"]["diagnostic_retrieval_preflight_rejections"] = (
            answerable_rejected + abstention_rejected
        )
        self._persist(run)
        return run

    def resume(self, run_id: str) -> dict[str, Any]:
        """Resume one run while preventing cross-process cursor corruption."""
        run_directory = self.output_directory / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        lock_path = run_directory / ".resume.lock"
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            try:
                owner = int(lock_path.read_text(encoding="utf-8").strip())
                os.kill(owner, 0)
            except (OSError, ValueError):
                lock_path.unlink(missing_ok=True)
                return self.resume(run_id)
            raise CurationSuspended(
                f"Autonomous run {run_id} is already active in process {owner}."
            ) from error
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        finally:
            os.close(descriptor)
        try:
            return self._resume_unlocked(run_id)
        finally:
            lock_path.unlink(missing_ok=True)

    def _resume_unlocked(self, run_id: str) -> dict[str, Any]:
        """Resume from the last persisted paper/question cursor."""
        run = self.get_run(run_id)
        if run.get("diagnostic", {}).get("valid_for_readiness") is False:
            return run
        if run["status"] == "completed":
            return run
        config = AutonomousCurationConfig.from_dict(run["config"])
        try:
            self._verify_sources(run)
            for paper_position in range(
                run["cursor"]["paper_index"], len(run["paper_ids"])
            ):
                paper_id = run["paper_ids"][paper_position]
                split = run["paper_splits"][paper_id]
                candidates = run["candidates"][paper_id]
                start = (
                    run["cursor"]["question_index"]
                    if paper_position == run["cursor"]["paper_index"]
                    else 0
                )
                accepted_for_paper = sum(
                    item.get("status") == "codex_curated"
                    and paper_id in item.get("paper_ids", [])
                    for item in run["records"]
                )
                for question_position in range(start, len(candidates)):
                    candidate = dict(candidates[question_position])
                    parent_id = candidate.get("parent_question_id")
                    if parent_id:
                        parent = next(
                            (
                                item
                                for item in run["records"]
                                if item.get("question_id") == parent_id
                                and item.get("status") == "codex_curated"
                            ),
                            None,
                        )
                        if parent is None:
                            record = {
                                "curation_record_id": _identifier("excluded"),
                                "question_id": candidate["question_id"],
                                "paper_ids": candidate["paper_ids"],
                                "question": candidate["question"],
                                "question_type": candidate["question_type"],
                                "status": "split_excluded",
                                "split": split,
                                "terminal_reasons": ["multi_turn_parent_not_curated"],
                            }
                            run["records"].append(record)
                            run["cursor"] = {
                                "paper_index": paper_position,
                                "question_index": question_position + 1,
                            }
                            self._persist(run)
                            continue
                        candidate["conversation_context"] = [
                            {"role": "user", "content": parent["question"]},
                            {
                                "role": "assistant",
                                "content": parent["answer"]["answer_text"],
                            },
                        ]
                    run["stage"] = "question_curation"
                    run["stage_id"] = f"{paper_id}:{candidate['question_id']}:curation"
                    candidate_splits = {
                        run["paper_splits"][item] for item in candidate["paper_ids"]
                    }
                    if len(candidate_splits) > 1:
                        record = {
                            "curation_record_id": _identifier("excluded"),
                            "question_id": candidate["question_id"],
                            "paper_ids": candidate["paper_ids"],
                            "question": candidate["question"],
                            "question_type": candidate["question_type"],
                            "status": "split_excluded",
                            "split": None,
                            "terminal_reasons": [
                                "cross_paper_sources_belong_to_different_splits"
                            ],
                        }
                    elif split == "test":
                        run["evaluation_questions"].append(candidate)
                        record = {
                            "curation_record_id": _identifier("excluded"),
                            "question_id": candidate["question_id"],
                            "paper_ids": [paper_id],
                            "question": candidate["question"],
                            "question_type": candidate["question_type"],
                            "status": "split_excluded",
                            "split": "test",
                            "terminal_reasons": [
                                "held_out_test_paper_no_answer_or_correction_stored"
                            ],
                        }
                    elif accepted_for_paper >= config.maximum_examples_per_paper:
                        break
                    else:
                        try:
                            record = self._curate_candidate(candidate, config)
                        except (RuntimeError, TypeError, ValueError) as error:
                            message = str(error)
                            if (
                                message.startswith("Codex ")
                                or "reviewer service" in message
                                or "provider returned an invalid result type" in message
                                or "Malformed Codex" in message
                            ):
                                raise
                            stage = (
                                "retrieval"
                                if "retriev" in message.casefold()
                                else "validation"
                                if "valid" in message.casefold()
                                else "construction"
                            )
                            signature = f"{type(error).__name__}:{stage}:{message}"
                            failures = run.setdefault("candidate_failures", [])
                            failures.append(
                                {
                                    "question_id": candidate["question_id"],
                                    "exception_class": type(error).__name__,
                                    "message": message,
                                    "stage": stage,
                                    "recoverable": True,
                                    "signature": signature,
                                }
                            )
                            record = {
                                "curation_record_id": _identifier("failed"),
                                "question_id": candidate["question_id"],
                                "paper_ids": candidate["paper_ids"],
                                "question": candidate["question"],
                                "question_type": candidate["question_type"],
                                "status": f"{stage}_failed",
                                "split": split,
                                "terminal_reasons": [message],
                                "failure": failures[-1],
                            }
                            repeated = sum(
                                item["signature"] == signature for item in failures
                            )
                            stage_fraction = sum(
                                item["stage"] == stage for item in failures
                            ) / max(1, len(run["records"]) + 1)
                            if (
                                repeated >= config.maximum_repeated_systemic_errors
                                or len(run["records"]) >= 3
                                and stage_fraction
                                > config.maximum_stage_failure_fraction
                            ):
                                raise CurationSuspended(
                                    f"systemic_candidate_failure:{signature}"
                                ) from error
                        record["split"] = split
                        record = self._apply_balance_policy(run, record)
                        if record["status"] == "codex_curated":
                            record["curation_example_id"] = self._materialize(record)
                            accepted_for_paper += 1
                    self._record_preflight_metrics(run, record)
                    run["records"].append(record)
                    run["cursor"] = {
                        "paper_index": paper_position,
                        "question_index": question_position + 1,
                    }
                    self._persist(run)
                    reason = self._stop_reason(run)
                    if reason is not None:
                        raise CurationSuspended(reason)
                run["cursor"] = {
                    "paper_index": paper_position + 1,
                    "question_index": 0,
                }
                self._persist(run)
        except CurationSuspended as error:
            run["status"] = "suspended"
            run["stage"] = (
                "codex_unavailable"
                if str(error) == "Codex reviewer service is unavailable."
                else "safety_stop"
            )
            run["errors"].append(str(error))
            self._persist(run)
            return run
        except (RuntimeError, TypeError, ValueError) as error:
            run["status"] = "suspended"
            run["stage"] = "reviewer_or_validation_error"
            run["errors"].append(f"{type(error).__name__}: {error}")
            self._persist(run)
            return run
        try:
            return self._complete(run)
        except (RuntimeError, TypeError, ValueError) as error:
            run["status"] = "suspended"
            run["stage"] = "dataset_export_error"
            run["errors"].append(f"{type(error).__name__}: {error}")
            self._persist(run)
            return run

    def _complete(self, run: dict[str, Any]) -> dict[str, Any]:
        run["report"] = autonomous_quality_report(run["records"])
        run["report"]["questions_generated"] = sum(
            len(items) for items in run["candidates"].values()
        )
        run["report"]["papers_processed"] = len(run["paper_ids"])
        run["report"]["test_evaluation_question_count"] = len(
            run["evaluation_questions"]
        )
        run_directory = self.output_directory / run["run_id"]
        run["status"] = "completed"
        run["stage"] = "report_complete"
        run["stage_id"] = _identifier("stage")
        accepted_ids = {
            item["curation_example_id"]
            for item in run["records"]
            if item.get("status") == "codex_curated"
        }
        examples = tuple(
            item
            for item in self.service._load_corrections()
            if item.example_id in accepted_ids
        )
        if examples:
            exported_papers = {
                paper_id for example in examples for paper_id in example.paper_ids
            }
            manual_splits = {
                paper_id: split
                for paper_id, split in run["paper_splits"].items()
                if split != "test" and paper_id in exported_papers
            }
            dataset = build_dataset(
                examples,
                dataset_version=__version__,
                seed=run["config"]["seed"],
                manual_paper_splits=manual_splits,
                trust_tier="codex-curated-only",
            )
            destination = run_directory / "codex_curated_dataset.json"
            save_dataset(dataset, destination)
            run["dataset_path"] = str(destination)
        manifest = {
            "run_id": run["run_id"],
            "package_version": __version__,
            "trust_class": "codex_curated",
            "human_reviewed": False,
            "paper_splits": run["paper_splits"],
            "source_hashes": run["source_hashes"],
            "analysis_hashes": run["analysis_hashes"],
            "config": run["config"],
            "report": run["report"],
            "dataset_path": run["dataset_path"],
            "evaluation_questions": run["evaluation_questions"],
            "record_hashes": [item.get("record_hash") for item in run["records"]],
        }
        manifest_path = run_directory / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        run["manifest_path"] = str(manifest_path)
        self._persist(run)
        return run

    def process_new(
        self, *, config: AutonomousCurationConfig | None = None
    ) -> dict[str, Any]:
        """Curate papers absent from every completed autonomous run."""
        run = self.create_new(config=config)
        return self.resume(run["run_id"])

    def create_new(
        self, *, config: AutonomousCurationConfig | None = None
    ) -> dict[str, Any]:
        """Create a run for indexed papers absent from completed runs."""
        completed = {
            paper_id
            for run in self._load_runs()
            if run.get("status") == "completed"
            for paper_id in run.get("paper_ids", [])
        }
        new_ids = tuple(
            item["document_id"]
            for item in self.service.list_papers()
            if item["document_id"] not in completed
        )
        if not new_ids:
            raise ValueError("No newly indexed papers require autonomous curation.")
        return self.create(paper_ids=new_ids, config=config)

    def export(self, run_id: str, *, trust_tier: str) -> dict[str, Any]:
        """Export selected trusted records from a completed autonomous run."""
        run = self.get_run(run_id)
        accepted_ids = {
            item["curation_example_id"]
            for item in run["records"]
            if item.get("status") == "codex_curated"
        }
        eligible_papers = {
            paper_id
            for paper_id, split in run["paper_splits"].items()
            if split != "test"
        }
        examples: tuple[GroundedInstructionExample, ...] = tuple(
            item
            for item in self.service._load_corrections()
            if item.example_id in accepted_ids
            or (
                trust_tier != "codex-curated-only"
                and item.review_status in {"human_approved", "codex_approved"}
                and set(item.paper_ids) <= eligible_papers
            )
        )
        exported_papers = {
            paper_id for example in examples for paper_id in example.paper_ids
        }
        dataset = build_dataset(
            examples,
            dataset_version=__version__,
            seed=run["config"]["seed"],
            manual_paper_splits={
                paper_id: split
                for paper_id, split in run["paper_splits"].items()
                if split != "test" and paper_id in exported_papers
            },
            trust_tier=trust_tier,
        )
        path = self.output_directory / run_id / f"dataset_{trust_tier}.json"
        save_dataset(dataset, path)
        return {"path": str(path), "example_count": len(dataset.examples)}

    def report(self, run_id: str) -> dict[str, Any]:
        """Return the measured quality report for one run."""
        run = self.get_run(run_id)
        return run.get("report") or autonomous_quality_report(run["records"])

    def corpus_metrics(self) -> dict[str, Any]:
        """Return compact dashboard metrics for autonomous curation."""
        runs = self._load_runs()
        records = [item for run in runs for item in run.get("records", [])]
        statuses = Counter(item.get("status", "unknown") for item in records)
        latest = runs[-1] if runs else None
        latest_summary = (
            None
            if latest is None
            else {
                key: latest.get(key)
                for key in (
                    "run_id",
                    "status",
                    "stage",
                    "stage_id",
                    "created_at",
                    "updated_at",
                    "paper_ids",
                    "paper_splits",
                    "cursor",
                    "report",
                    "dataset_path",
                    "manifest_path",
                    "errors",
                    "config",
                )
            }
        )
        if latest_summary is not None and latest_summary.get("report") is None:
            latest_summary["report"] = autonomous_quality_report(
                latest.get("records", [])
            )
        if latest_summary is not None and latest is not None:
            latest_summary["claim_diagnostics"] = [
                {
                    "question_id": item.get("question_id"),
                    "question": item.get("question"),
                    "question_type": item.get("question_type"),
                    "supported_claim_graph": item.get("supported_claim_graph", {}),
                    "claim_alignment_metrics": item.get("claim_alignment_metrics", {}),
                    "repair_history": item.get("repair_history", []),
                    "claim_level_disagreements": item.get(
                        "claim_level_disagreements", []
                    ),
                    "evidence": item.get("answer", {}).get("evidence", []),
                }
                for item in latest.get("records", [])[-10:]
            ]
        return {
            "run_count": len(runs),
            "active_run_count": sum(
                item.get("status") in {"running", "suspended"} for item in runs
            ),
            "status_counts": dict(sorted(statuses.items())),
            "latest_run": latest_summary,
            "codex_available": self.service.codex_provider.available(),
            "codex_identity": list(self.service.codex_provider.identity),
            "storage": str(self.output_directory),
        }
