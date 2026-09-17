from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from payrag.retrieval import SearchResult


DECISIONS = {
    "answerable",
    "needs_clarification",
    "insufficient_evidence",
    "needs_runtime_evidence",
    "out_of_scope",
}


class AnswerContractError(ValueError):
    """Raised when a generated answer violates the answer contract."""


def prepare_retrieval_context(
    index: Any,
    manifest: dict[str, Any],
    query: str,
    *,
    top_k: int = 5,
    max_per_source: int | None = None,
    retriever_kind: str = "bm25",
) -> dict[str, Any]:
    urls = {
        source["source_id"]: source["canonical_url"] for source in manifest["sources"]
    }
    results = index.search(
        query, top_k=top_k, max_per_source=max_per_source
    )
    return {
        "question": query,
        "retrieval": {
            "kind": retriever_kind,
            "top_k": top_k,
            "max_per_source": max_per_source,
        },
        "evidence": [_context_record(result, urls) for result in results],
        "generation_status": "not_run",
        "answer_contract": {
            "decision": sorted(DECISIONS),
            "required_fields": [
                "decision",
                "applicable_conditions",
                "conclusions",
                "missing_information",
                "verification_steps",
                "operational_error",
            ],
            "citation_rule": (
                "Each conclusion must cite one or more chunk IDs from this evidence set, "
                "unless it is explicitly marked as an engineering inference."
            ),
        },
    }


def validate_answer_contract(
    answer: dict[str, Any], allowed_chunk_ids: set[str]
) -> None:
    required_fields = {
        "decision",
        "applicable_conditions",
        "conclusions",
        "missing_information",
        "verification_steps",
        "operational_error",
    }
    missing = required_fields - answer.keys()
    if missing:
        raise AnswerContractError(
            f"Missing answer fields: {', '.join(sorted(missing))}"
        )

    decision = answer["decision"]
    operational_error = answer["operational_error"]
    if decision not in DECISIONS:
        raise AnswerContractError(f"Unknown decision: {decision}")
    if operational_error is not None and not isinstance(operational_error, dict):
        raise AnswerContractError("operational_error must be null or an object")

    for field in ("applicable_conditions", "conclusions", "missing_information", "verification_steps"):
        if not isinstance(answer[field], list):
            raise AnswerContractError(f"{field} must be a list")

    for index, conclusion in enumerate(answer["conclusions"]):
        if not isinstance(conclusion, dict) or not isinstance(
            conclusion.get("statement"), str
        ):
            raise AnswerContractError(f"conclusions[{index}] requires a statement")
        citations = conclusion.get("evidence_chunk_ids", [])
        if not isinstance(citations, list) or not all(
            isinstance(citation, str) for citation in citations
        ):
            raise AnswerContractError(
                f"conclusions[{index}].evidence_chunk_ids must be a string list"
            )
        unknown = set(citations) - allowed_chunk_ids
        if unknown:
            raise AnswerContractError(
                f"conclusions[{index}] cites unavailable chunks: {', '.join(sorted(unknown))}"
            )
        engineering_inference = conclusion.get("engineering_inference", False)
        if not isinstance(engineering_inference, bool):
            raise AnswerContractError(
                f"conclusions[{index}].engineering_inference must be boolean"
            )
        if not citations and not engineering_inference:
            raise AnswerContractError(
                f"conclusions[{index}] needs evidence or an engineering inference marker"
            )


def write_context(path: Path, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _context_record(
    result: SearchResult, urls: dict[str, str]
) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "source_id": result.source_id,
        "chunk_id": result.chunk_id,
        "section_path": result.section_path,
        "canonical_url": urls[result.source_id],
        "text": result.text,
        "parent_chunk_id": result.parent_chunk_id,
    }
