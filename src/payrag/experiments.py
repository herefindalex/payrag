from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml

from payrag.retrieval import BM25Index, SearchResult


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_per_source: int | None = None,
    ) -> list[SearchResult]: ...


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_pilot(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("pilot file must contain a cases list")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_evidence_binding_candidates(
    *,
    snapshot_id: str,
    pilot_path: Path,
    chunks_path: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Rank locator candidates without mutating the human-owned pilot gold."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    pilot = _read_pilot(pilot_path)
    chunks = _read_jsonl(chunks_path)
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk["source_id"]].append(chunk)

    source_indexes = {
        source_id: BM25Index(source_chunks)
        for source_id, source_chunks in chunks_by_source.items()
    }
    case_records: list[dict[str, Any]] = []
    candidate_total = 0
    locator_total = 0

    for case in pilot["cases"]:
        bindings: list[dict[str, Any]] = []
        for ordinal, evidence in enumerate(case["evidence"]):
            locator_total += 1
            source_id = evidence["source_id"]
            locator = evidence["section_or_locator"]
            index = source_indexes.get(source_id)
            results = [] if index is None else index.search(locator, top_k=top_k)
            candidates = [
                {
                    "rank": result.rank,
                    "score": round(result.score, 6),
                    "chunk_id": result.chunk_id,
                    "section_path": result.section_path,
                    "excerpt": result.text[:400],
                }
                for result in results
            ]
            candidate_total += len(candidates)
            bindings.append(
                {
                    "evidence_ordinal": ordinal,
                    "source_id": source_id,
                    "role": evidence["role"],
                    "section_or_locator": locator,
                    "current_snapshot_chunk_ids": evidence.get(
                        "snapshot_chunk_ids", []
                    ),
                    "review_status": "pending_human_review",
                    "candidates": candidates,
                }
            )
        case_records.append({"case_id": case["case_id"], "bindings": bindings})

    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at": _utc_now(),
        "status": "pending_human_review",
        "inputs": {
            "pilot_path": str(pilot_path),
            "chunks_path": str(chunks_path),
            "pilot_dataset_version": pilot.get("dataset_version"),
        },
        "method": {
            "name": "source_scoped_bm25_locator_candidates",
            "top_k": top_k,
            "query_field": "evidence.section_or_locator",
            "source_filter": "evidence.source_id",
        },
        "guardrails": {
            "writes_pilot_gold": False,
            "candidate_ids_are_gold": False,
            "required_action": (
                "A human reviewer must inspect the snapshot text and explicitly "
                "accept every required, qualifier, and alternative binding."
            ),
        },
        "summary": {
            "case_count": len(case_records),
            "evidence_locator_count": locator_total,
            "candidate_count": candidate_total,
        },
        "cases": case_records,
    }


def _source_urls(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        source["source_id"]: source["canonical_url"]
        for source in payload["sources"]
    }


def _retrieval_record(
    result: SearchResult, source_urls: dict[str, str]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": result.rank,
        "score": round(result.score, 6),
        "source_id": result.source_id,
        "chunk_id": result.chunk_id,
        "section_path": result.section_path,
        "canonical_url": source_urls[result.source_id],
        "text": result.text,
    }
    if result.parent_chunk_id is not None:
        record["parent_chunk_id"] = result.parent_chunk_id
    return record


def _context_variant(
    retriever: Retriever,
    query: str,
    source_urls: dict[str, str],
    *,
    top_k: int,
    max_per_source: int | None,
) -> dict[str, Any]:
    results = retriever.search(
        query, top_k=top_k, max_per_source=max_per_source
    )
    evidence = [_retrieval_record(result, source_urls) for result in results]
    return {
        "status": "prepared_generation_provider_not_configured",
        "evidence": evidence,
        "observed_character_count": sum(len(item["text"]) for item in evidence),
        "observed_whitespace_token_count": sum(
            len(item["text"].split()) for item in evidence
        ),
        "generation_token_count": None,
    }


def build_m2_context_manifest(
    *,
    snapshot_id: str,
    pilot_path: Path,
    source_manifest_path: Path,
    bm25: Retriever,
    dense: Retriever,
    evidence_candidates_path: Path,
    top_k: int = 5,
    max_per_source: int | None = 1,
) -> dict[str, Any]:
    pilot = _read_pilot(pilot_path)
    source_urls = _source_urls(source_manifest_path)
    candidate_report = json.loads(
        evidence_candidates_path.read_text(encoding="utf-8")
    )
    if candidate_report.get("snapshot_id") != snapshot_id:
        raise ValueError("candidate report snapshot does not match requested snapshot")
    if candidate_report.get("status") != "pending_human_review":
        raise ValueError("unexpected candidate report review status")

    cases: list[dict[str, Any]] = []
    for case in pilot["cases"]:
        query = case["prompt_en"]
        cases.append(
            {
                "case_id": case["case_id"],
                "query_field": "prompt_en",
                "query": query,
                "variants": {
                    "no_context": {
                        "status": "prepared_generation_provider_not_configured",
                        "evidence": [],
                        "observed_character_count": 0,
                        "observed_whitespace_token_count": 0,
                        "generation_token_count": 0,
                    },
                    "bm25": _context_variant(
                        bm25,
                        query,
                        source_urls,
                        top_k=top_k,
                        max_per_source=max_per_source,
                    ),
                    "dense": _context_variant(
                        dense,
                        query,
                        source_urls,
                        top_k=top_k,
                        max_per_source=max_per_source,
                    ),
                    "oracle": {
                        "status": "blocked_pending_human_review",
                        "evidence": [],
                        "blocked_by": "snapshot_chunk_ids_not_accepted_as_gold",
                    },
                    "evidence_ablation": {
                        "status": "blocked_pending_human_review",
                        "evidence": [],
                        "blocked_by": "oracle_and_alternative_evidence_not_accepted",
                        "planned_rule": (
                            "Remove every human-accepted supporting chunk and all "
                            "accepted alternatives from the comparison context."
                        ),
                    },
                },
            }
        )

    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at": _utc_now(),
        "status": "contexts_prepared_generation_and_gold_review_pending",
        "inputs": {
            "pilot_path": str(pilot_path),
            "source_manifest_path": str(source_manifest_path),
            "evidence_candidates_path": str(evidence_candidates_path),
        },
        "comparison_controls": {
            "same_snapshot": True,
            "query_field": "prompt_en",
            "top_k": top_k,
            "max_per_source": max_per_source,
            "generation_model": None,
            "generation_prompt_version": None,
            "max_context_tokens": None,
            "actual_generation_usage": None,
            "claim_eligible": False,
        },
        "blocking_conditions": [
            "generation_provider_not_configured",
            "generation_model_and_tokenizer_not_selected",
            "maximum_context_token_budget_not_fixed",
            "snapshot_gold_evidence_pending_human_review",
        ],
        "variant_definitions": {
            "no_context": "Question only.",
            "bm25": "BM25 results with at most one result per source.",
            "dense": "MiniLM cosine results with at most one result per source.",
            "oracle": "Human-accepted complete supporting evidence.",
            "evidence_ablation": (
                "A comparison context after all accepted support and alternatives "
                "have been removed."
            ),
        },
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PayRAG experiment artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates = subparsers.add_parser("evidence-candidates")
    candidates.add_argument("--snapshot-id", required=True)
    candidates.add_argument("--pilot", type=Path, required=True)
    candidates.add_argument("--chunks", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True)
    candidates.add_argument("--top-k", type=int, default=5)

    contexts = subparsers.add_parser("m2-contexts")
    contexts.add_argument("--snapshot-id", required=True)
    contexts.add_argument("--pilot", type=Path, required=True)
    contexts.add_argument("--source-manifest", type=Path, required=True)
    contexts.add_argument("--bm25-index", type=Path, required=True)
    contexts.add_argument("--dense-index", type=Path, required=True)
    contexts.add_argument("--evidence-candidates", type=Path, required=True)
    contexts.add_argument("--output", type=Path, required=True)
    contexts.add_argument("--top-k", type=int, default=5)
    contexts.add_argument("--max-per-source", type=int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evidence-candidates":
        payload = build_evidence_binding_candidates(
            snapshot_id=args.snapshot_id,
            pilot_path=args.pilot,
            chunks_path=args.chunks,
            top_k=args.top_k,
        )
    else:
        from payrag.dense import DenseIndex

        payload = build_m2_context_manifest(
            snapshot_id=args.snapshot_id,
            pilot_path=args.pilot,
            source_manifest_path=args.source_manifest,
            bm25=BM25Index.load(args.bm25_index),
            dense=DenseIndex.load(args.dense_index),
            evidence_candidates_path=args.evidence_candidates,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
        )
    _write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
