"""Command-line interface for evidence-controlled local answering."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from localml_scholar.answering.evidence import EvidenceSelectionConfig
from localml_scholar.answering.generative import (
    GroundedGenerationConfig,
    GroundedGenerativeAnswerer,
)
from localml_scholar.answering.models import GroundedAnswer
from localml_scholar.answering.pipeline import GroundedAnswerPipeline
from localml_scholar.answering.serialization import (
    answer_artifact_state,
    save_grounded_answer,
)
from localml_scholar.retrieval import RetrievalIndex, SearchFilters

_CLI_METHODS = {
    "top-passage": "top_passage",
    "extractive": "extractive",
    "generative": "generative",
    "generative-with-extractive-fallback": ("generative_with_extractive_fallback"),
}


def _filters(args: argparse.Namespace) -> SearchFilters:
    return SearchFilters(
        document_id=args.document_id,
        source_name=args.source_name,
        media_type=args.media_type,
        heading_path_prefix=tuple(args.heading_prefix or ()),
        publication_year=args.publication_year,
        logical_collection=args.collection,
    )


def _generative_answerer(
    args: argparse.Namespace,
) -> GroundedGenerativeAnswerer | None:
    method = _CLI_METHODS[args.method]
    if method not in {
        "generative",
        "generative_with_extractive_fallback",
    }:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is only valid for generative methods.")
        return None
    if args.checkpoint is None:
        raise ValueError(
            f"The {args.method} method requires --checkpoint with a local "
            "transformer model and matching tokenizer."
        )
    config = GroundedGenerationConfig(
        maximum_new_tokens=args.maximum_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.sampling_top_k,
        seed=args.seed,
        decoded_stop_delimiter=args.stop_delimiter,
        decode_errors=args.decode_errors,
    )
    return GroundedGenerativeAnswerer.from_checkpoint(
        args.checkpoint,
        config=config,
    )


def run_answer(args: argparse.Namespace) -> GroundedAnswer:
    """Load explicit local artifacts and execute one answer request."""
    index = RetrievalIndex.load(args.index)
    pipeline = GroundedAnswerPipeline(
        index,
        evidence_config=EvidenceSelectionConfig(
            retrieval_method=args.retriever,
            retrieval_top_k=args.top_k,
            evidence_top_k=args.evidence_top_k,
            maximum_evidence_characters=args.maximum_evidence_characters,
        ),
        generative_answerer=_generative_answerer(args),
    )
    answer = pipeline.answer(
        args.question,
        method=_CLI_METHODS[args.method],
        filters=_filters(args),
    )
    if args.save is not None:
        save_grounded_answer(args.save, answer)
    return answer


def _json_payload(answer: GroundedAnswer, args: argparse.Namespace) -> dict[str, Any]:
    payload = answer_artifact_state(answer)
    payload["operation"] = "answer"
    payload["saved_artifact"] = None if args.save is None else str(args.save)
    return payload


def _human_readable(
    answer: GroundedAnswer,
    *,
    verbose: bool,
    saved_artifact: Path | None,
) -> str:
    status = "abstained" if answer.abstained else "answered"
    if answer.fallback_used:
        status = "extractive fallback"
    lines = [
        f"question: {answer.question}",
        f"method: {answer.method}",
        f"status: {status}",
        "",
        "answer:",
        answer.answer_text,
        "",
        "validation:",
        f"  accepted: {answer.validation.accepted}",
        f"  citations valid: {answer.validation.citations_valid}",
        f"  citation coverage: {answer.validation.citation_coverage:.6f}",
        f"  unsupported claims: {answer.validation.unsupported_claim_count}",
        "  rejection reasons: "
        f"{', '.join(answer.validation.rejection_reasons) or '(none)'}",
        "",
        "sources:",
    ]
    for item in answer.evidence:
        lines.append(
            f"  {item.label}: {item.citation.format()} "
            f"(score={item.retrieval_score:.12f})"
        )
        if verbose:
            lines.extend(["    passage:", item.selected_text])
    if not answer.evidence:
        lines.append("  (none)")
    if answer.abstention_reason is not None:
        lines.append(f"abstention reason: {answer.abstention_reason}")
    if answer.fallback_reason is not None:
        lines.append(f"fallback reason: {answer.fallback_reason}")
    if answer.raw_generated_text is not None:
        lines.extend(["", "raw model generation:", answer.raw_generated_text])
    if saved_artifact is not None:
        lines.append(f"saved artifact: {saved_artifact}")
    return "\n".join(lines)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone grounded-answer command."""
    parser = argparse.ArgumentParser(
        description="Answer only from a deterministic local retrieval index."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--method",
        choices=tuple(_CLI_METHODS),
        default="extractive",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf"),
        default="bm25",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--evidence-top-k", type=int, default=4)
    parser.add_argument("--maximum-evidence-characters", type=int, default=4000)
    parser.add_argument("--document-id")
    parser.add_argument("--source-name")
    parser.add_argument("--media-type")
    parser.add_argument("--heading-prefix", action="append")
    parser.add_argument("--publication-year", type=int)
    parser.add_argument("--collection")
    parser.add_argument("--checkpoint", type=Path)
    generation = parser.add_mutually_exclusive_group()
    generation.add_argument("--greedy", action="store_true", default=True)
    generation.add_argument("--sample", dest="greedy", action="store_false")
    parser.add_argument("--maximum-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-delimiter")
    parser.add_argument(
        "--decode-errors",
        choices=("strict", "replace"),
        default="replace",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one grounded answer and print its full validation state."""
    args = parse_args(arguments)
    answer = run_answer(args)
    if args.json:
        print(
            json.dumps(
                _json_payload(answer, args),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            _human_readable(
                answer,
                verbose=args.verbose,
                saved_artifact=args.save,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
