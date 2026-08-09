"""Resumable application orchestration for fully automated corpus curation."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from localml_scholar._version import __version__
from localml_scholar.review_app.storage import (
    atomic_write_json,
    load_json_list,
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
    balanced_paper_splits,
    build_dataset,
    generate_paper_questions,
    propose_correction,
    review_interaction_second_pass,
    save_dataset,
)
from localml_scholar.training_data.provenance import content_sha256

if TYPE_CHECKING:
    from localml_scholar.review_app.service import ReviewService


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


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
        for paper_id in selected:
            _index, document = self.service._document(paper_id)
            source_hashes[paper_id] = content_sha256(document.to_dict())
            analysis = self.service.analyze(paper_id)
            analysis_hashes[paper_id] = content_sha256(
                {
                    "analysis": analysis["analysis"],
                    "summary": analysis["summary"],
                    "checklist": analysis["checklist"],
                }
            )
            generated = generate_paper_questions(
                paper_id,
                document.title or document.source_name,
                count=80,
                section_titles=tuple(
                    section.heading or "Untitled section"
                    for section in document.sections
                ),
            )
            eligible = [
                item.to_dict()
                for item in generated
                if self._candidate_enabled(item.question_type, config)
            ]
            candidates[paper_id] = eligible[: config.questions_per_paper]
            section_candidates = []
            for section in document.sections[:10]:
                heading = section.heading or "Untitled section"
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
            if section_candidates:
                keep = max(0, config.questions_per_paper - len(section_candidates))
                candidates[paper_id] = (
                    candidates[paper_id][:keep] + section_candidates
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
            "stage": "paper_analysis_complete",
            "stage_id": _identifier("stage"),
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "config": config.to_dict(),
            "paper_ids": list(selected),
            "paper_splits": splits,
            "source_hashes": source_hashes,
            "analysis_hashes": analysis_hashes,
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

    def _verify_sources(self, run: dict[str, Any]) -> None:
        for paper_id, expected in run["source_hashes"].items():
            _index, document = self.service._document(paper_id)
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
        )
        interaction["question_id"] = candidate_state["question_id"]
        interaction["question_type"] = candidate_state["question_type"]
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
            method = "hybrid" if attempt == 1 else "bm25"
            expected = candidate_state.get("expected_sections", [])
            heading = (str(expected[0]),) if attempt == 1 and expected else None
            results = self.service.search_evidence(
                query=expanded,
                paper_ids=selected,
                top_k=10,
                method=method,
                heading_path_prefix=heading,
            )
            if not results and heading is not None:
                results = self.service.search_evidence(
                    query=expanded,
                    paper_ids=selected,
                    top_k=10,
                    method=method,
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

    def _apply_balance_policy(
        self, run: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        if record["status"] != "codex_curated":
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
        disagreement = sum(bool(item.get("reviewer_disagreement")) for item in records)
        if disagreement / len(records) > run["config"]["maximum_disagreement_rate"]:
            return "Reviewer disagreement exceeded the configured safety threshold."
        malformed = sum(
            "reviewer_error" in item.get("terminal_reasons", []) for item in records
        )
        if malformed >= 3:
            return "Repeated malformed reviewer output triggered a safety stop."
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
            citation_failures = sum(
                "citation_validation_failed" in item.get("terminal_reasons", [])
                for item in reviewed
            )
            if citation_failures / len(reviewed) > 0.50:
                return "Citation validation failed systematically."
        duplicate_rate = sum(
            item.get("status") == "duplicate" for item in records
        ) / len(records)
        if duplicate_rate > 0.50:
            return "Excessive duplicate generation triggered a safety stop."
        return None

    def resume(self, run_id: str) -> dict[str, Any]:
        """Resume from the last persisted paper/question cursor."""
        run = self.get_run(run_id)
        if run["status"] == "completed":
            return run
        if not self.service.codex_provider.available():
            run["status"] = "suspended"
            run["stage"] = "codex_unavailable"
            run["errors"].append("Codex reviewer service is unavailable.")
            self._persist(run)
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
                        record = self._curate_candidate(candidate, config)
                        record["split"] = split
                        record = self._apply_balance_policy(run, record)
                        if record["status"] == "codex_curated":
                            record["curation_example_id"] = self._materialize(record)
                            accepted_for_paper += 1
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
            run["stage"] = "safety_stop"
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
            manual_splits = {
                paper_id: split
                for paper_id, split in run["paper_splits"].items()
                if split != "test"
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
        dataset = build_dataset(
            examples,
            dataset_version=__version__,
            seed=run["config"]["seed"],
            manual_paper_splits={
                paper_id: split
                for paper_id, split in run["paper_splits"].items()
                if split != "test"
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
