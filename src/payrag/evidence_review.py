from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from payrag.retrieval import BM25Index


ATTESTATION = "I inspected the snapshot text for every accepted chunk."


class EvidenceReviewError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceReviewError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceReviewError(f"{path} must contain a YAML object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_evidence_review(
    *,
    pilot_path: Path,
    chunks_path: Path,
    candidates_path: Path,
    proposal_overrides_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    pilot = _read_yaml(pilot_path)
    candidates = _read_json(candidates_path)
    chunk_records = _read_jsonl(chunks_path)
    chunks = {chunk["chunk_id"]: chunk for chunk in chunk_records}
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunk_records:
        chunks_by_source.setdefault(chunk["source_id"], []).append(chunk)
    source_indexes = {
        source_id: BM25Index(source_chunks)
        for source_id, source_chunks in chunks_by_source.items()
    }
    pilot_cases = {case["case_id"]: case for case in pilot.get("cases", [])}
    if candidates.get("status") != "pending_human_review":
        raise EvidenceReviewError("candidate report must be pending_human_review")

    overrides: dict[tuple[str, int], dict[str, Any]] = {}
    overrides_sha256 = None
    if proposal_overrides_path is not None:
        override_payload = _read_yaml(proposal_overrides_path)
        if override_payload.get("snapshot_id") != candidates["snapshot_id"]:
            raise EvidenceReviewError("proposal override snapshot does not match")
        override_entries = override_payload.get("bindings", [])
        overrides = {
            (entry["case_id"], entry["evidence_ordinal"]): entry
            for entry in override_entries
        }
        if len(overrides) != len(override_entries):
            raise EvidenceReviewError("proposal overrides contain duplicate bindings")
        overrides_sha256 = _sha256(proposal_overrides_path)

    decisions: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_id": candidates["snapshot_id"],
        "candidate_report_sha256": _sha256(candidates_path),
        "status": "pending_human_review",
        "reviewer": None,
        "reviewed_at": None,
        "attestation": None,
        "proposal_overrides_sha256": overrides_sha256,
        "cases": [],
    }
    lines = [
        f"# Evidence review — snapshot {candidates['snapshot_id']}",
        "",
        "Review every binding against the full normalized snapshot text below.",
        "Each binding starts with a system proposal. Candidate rank and proposal",
        "are review aids rather than approval decisions. Set `review_action` to",
        "`accept_proposal` or `accept_modified`, then add the reviewer, UTC time,",
        "top-level `approved`",
        f"status, and this exact attestation: `{ATTESTATION}`",
        "",
    ]

    for case_record in candidates.get("cases", []):
        case_id = case_record["case_id"]
        pilot_case = pilot_cases.get(case_id)
        if pilot_case is None:
            raise EvidenceReviewError(f"candidate case {case_id} is absent from pilot")
        lines.extend(
            [
                f"## {case_id}",
                "",
                f"**English prompt:** {pilot_case.get('prompt_en', '')}",
                "",
                f"**Traditional Chinese prompt:** {pilot_case.get('prompt_zh_TW', '')}",
                "",
            ]
        )
        decision_case = {"case_id": case_id, "bindings": []}
        for binding in case_record.get("bindings", []):
            ordinal = binding["evidence_ordinal"]
            source_id = binding["source_id"]
            source_index = source_indexes.get(source_id)
            if source_index is None:
                raise EvidenceReviewError(f"source {source_id} has no snapshot chunks")
            proposal_query = " ".join(
                part
                for part in (
                    pilot_case.get("prompt_en", ""),
                    binding["section_or_locator"],
                )
                if part
            )
            proposal_results = source_index.search(proposal_query, top_k=3)
            proposal = {
                "origin": "source_scoped_bm25_prompt_plus_locator",
                "confidence": "uncalibrated",
                "proposed_chunk_ids": [result.chunk_id for result in proposal_results],
                "rationale": (
                    "Top same-source chunks for the English prompt plus evidence "
                    "locator. Review the full text before accepting."
                ),
            }
            override = overrides.get((case_id, ordinal))
            if override is not None:
                proposed_ids = override.get("proposed_chunk_ids")
                if not isinstance(proposed_ids, list) or not proposed_ids:
                    raise EvidenceReviewError(
                        f"proposal override {(case_id, ordinal)} needs chunk IDs"
                    )
                proposal = {
                    "origin": "reviewer_seeded_override",
                    "confidence": override.get("confidence", "unreviewed"),
                    "proposed_chunk_ids": proposed_ids,
                    "rationale": override.get("rationale", ""),
                }
            for chunk_id in proposal["proposed_chunk_ids"]:
                chunk = chunks.get(chunk_id)
                if chunk is None or chunk["source_id"] != source_id:
                    raise EvidenceReviewError(
                        f"proposal chunk {chunk_id} is absent or has the wrong source"
                    )
            lines.extend(
                [
                    f"### Binding {ordinal}: {source_id} — {binding['role']}",
                    "",
                    f"Locator: `{binding['section_or_locator']}`",
                    "",
                    f"System proposal ({proposal['origin']}): "
                    + ", ".join(
                        f"`{chunk_id}`"
                        for chunk_id in proposal["proposed_chunk_ids"]
                    ),
                    "",
                    f"Proposal rationale: {proposal['rationale']}",
                    "",
                ]
            )
            candidate_ids: list[str] = []
            display_candidates = list(binding.get("candidates", []))
            displayed_ids = {
                candidate["chunk_id"] for candidate in display_candidates
            }
            for result in proposal_results:
                if result.chunk_id not in displayed_ids:
                    display_candidates.append(
                        {
                            "rank": "proposal",
                            "score": round(result.score, 6),
                            "chunk_id": result.chunk_id,
                        }
                    )
                    displayed_ids.add(result.chunk_id)
            for chunk_id in proposal["proposed_chunk_ids"]:
                if chunk_id not in displayed_ids:
                    display_candidates.append(
                        {"rank": "override", "score": "n/a", "chunk_id": chunk_id}
                    )
                    displayed_ids.add(chunk_id)
            for candidate in display_candidates:
                chunk_id = candidate["chunk_id"]
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    raise EvidenceReviewError(f"candidate chunk {chunk_id} is absent")
                if chunk["source_id"] != source_id:
                    raise EvidenceReviewError(
                        f"candidate chunk {chunk_id} does not belong to {source_id}"
                    )
                candidate_ids.append(chunk_id)
                section = " > ".join(chunk.get("section_path", [])) or "(root)"
                lines.extend(
                    [
                        (
                            f"#### Rank {candidate['rank']} · `{chunk_id}` · "
                            f"score {candidate['score']}"
                        ),
                        "",
                        f"Section: {section}",
                        "",
                        "<details><summary>Full normalized chunk text</summary>",
                        "",
                        f"<pre>{html.escape(chunk['text'])}</pre>",
                        "",
                        "</details>",
                        "",
                    ]
                )
            decision_case["bindings"].append(
                {
                    "evidence_ordinal": ordinal,
                    "source_id": source_id,
                    "role": binding["role"],
                    "section_or_locator": binding["section_or_locator"],
                    "candidate_chunk_ids": candidate_ids,
                    "proposal": proposal,
                    "review_action": "pending_human_review",
                    "accepted_chunk_ids": [],
                    "notes": "",
                }
            )
        decisions["cases"].append(decision_case)

    expected_override_keys = {
        (case["case_id"], binding["evidence_ordinal"])
        for case in candidates.get("cases", [])
        for binding in case.get("bindings", [])
    }
    unknown_overrides = set(overrides) - expected_override_keys
    if unknown_overrides:
        raise EvidenceReviewError(
            f"proposal overrides contain unknown bindings: {sorted(unknown_overrides)}"
        )

    return "\n".join(lines).rstrip() + "\n", decisions


def validate_evidence_review(
    *,
    decisions_path: Path,
    chunks_path: Path,
    candidates_path: Path,
) -> dict[str, Any]:
    decisions = _read_yaml(decisions_path)
    candidates = _read_json(candidates_path)
    chunks = {chunk["chunk_id"]: chunk for chunk in _read_jsonl(chunks_path)}
    if candidates.get("status") != "pending_human_review":
        raise EvidenceReviewError("candidate report must be pending_human_review")
    if decisions.get("snapshot_id") != candidates.get("snapshot_id"):
        raise EvidenceReviewError("decision and candidate snapshots differ")
    if decisions.get("candidate_report_sha256") != _sha256(candidates_path):
        raise EvidenceReviewError("candidate report hash does not match decision file")
    if decisions.get("status") != "approved":
        raise EvidenceReviewError("top-level review status must be approved")
    reviewer = decisions.get("reviewer")
    reviewed_at = decisions.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise EvidenceReviewError("reviewer is required")
    if not isinstance(reviewed_at, str):
        raise EvidenceReviewError("reviewer and reviewed_at are required")
    try:
        review_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceReviewError("reviewed_at must be an ISO 8601 timestamp") from exc
    if review_time.tzinfo is None or review_time.utcoffset() != timezone.utc.utcoffset(None):
        raise EvidenceReviewError("reviewed_at must include a UTC offset")
    if decisions.get("attestation") != ATTESTATION:
        raise EvidenceReviewError("review attestation is missing or does not match")

    expected_entries = [
        ((case["case_id"], binding["evidence_ordinal"]), binding)
        for case in candidates.get("cases", [])
        for binding in case.get("bindings", [])
    ]
    expected = dict(expected_entries)
    if not expected:
        raise EvidenceReviewError("candidate report contains no review bindings")
    if len(expected) != len(expected_entries):
        raise EvidenceReviewError("candidate report has duplicate binding identifiers")
    reviewed_entries = [
        ((case["case_id"], binding["evidence_ordinal"]), binding)
        for case in decisions.get("cases", [])
        for binding in case.get("bindings", [])
    ]
    reviewed = dict(reviewed_entries)
    if len(reviewed) != len(reviewed_entries):
        raise EvidenceReviewError("review contains duplicate binding identifiers")
    if set(reviewed) != set(expected):
        raise EvidenceReviewError("review bindings do not exactly match candidates")

    bindings: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    for key, expected_binding in expected.items():
        binding = reviewed[key]
        for field in ("source_id", "role", "section_or_locator"):
            if binding.get(field) != expected_binding.get(field):
                raise EvidenceReviewError(f"binding {key} changed protected field {field}")
        review_action = binding.get("review_action")
        if review_action == "accept_proposal":
            proposal = binding.get("proposal")
            if not isinstance(proposal, dict):
                raise EvidenceReviewError(f"binding {key} has no proposal")
            selected = proposal.get("proposed_chunk_ids")
        elif review_action == "accept_modified":
            selected = binding.get("accepted_chunk_ids")
            if not str(binding.get("notes", "")).strip():
                raise EvidenceReviewError(
                    f"binding {key} needs notes for an accept_modified decision"
                )
        else:
            raise EvidenceReviewError(
                f"binding {key} review_action must be accept_proposal or accept_modified"
            )
        if not isinstance(selected, list) or not selected:
            raise EvidenceReviewError(f"binding {key} needs accepted chunk IDs")
        if any(not isinstance(chunk_id, str) for chunk_id in selected):
            raise EvidenceReviewError(f"binding {key} has a non-string chunk ID")
        if len(selected) != len(set(selected)):
            raise EvidenceReviewError(f"binding {key} contains duplicate chunk IDs")
        candidate_ids = {
            candidate["chunk_id"]
            for candidate in expected_binding.get("candidates", [])
        }
        selection_records = []
        for chunk_id in selected:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise EvidenceReviewError(f"accepted chunk {chunk_id} is absent")
            if chunk["source_id"] != binding["source_id"]:
                raise EvidenceReviewError(
                    f"accepted chunk {chunk_id} has the wrong source"
                )
            accepted_ids.add(chunk_id)
            selection_records.append(
                {
                    "chunk_id": chunk_id,
                    "was_ranked_candidate": chunk_id in candidate_ids,
                    "section_path": chunk.get("section_path", []),
                }
            )
        bindings.append(
            {
                "case_id": key[0],
                "evidence_ordinal": key[1],
                "source_id": binding["source_id"],
                "role": binding["role"],
                "section_or_locator": binding["section_or_locator"],
                "review_action": review_action,
                "accepted_chunks": selection_records,
                "notes": binding.get("notes", ""),
            }
        )

    return {
        "schema_version": 1,
        "snapshot_id": decisions["snapshot_id"],
        "status": "approved_human_review",
        "reviewer": decisions["reviewer"],
        "reviewed_at": decisions["reviewed_at"],
        "attestation": decisions["attestation"],
        "inputs": {
            "candidate_report_sha256": decisions["candidate_report_sha256"],
            "decisions_sha256": _sha256(decisions_path),
        },
        "summary": {
            "binding_count": len(bindings),
            "unique_accepted_chunk_count": len(accepted_ids),
        },
        "bindings": bindings,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate gold review")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--pilot", type=Path, required=True)
    prepare.add_argument("--chunks", type=Path, required=True)
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--proposal-overrides", type=Path)
    prepare.add_argument("--packet-output", type=Path, required=True)
    prepare.add_argument("--decisions-output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--decisions", type=Path, required=True)
    validate.add_argument("--chunks", type=Path, required=True)
    validate.add_argument("--candidates", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            packet, decisions = prepare_evidence_review(
                pilot_path=args.pilot,
                chunks_path=args.chunks,
                candidates_path=args.candidates,
                proposal_overrides_path=args.proposal_overrides,
            )
            args.packet_output.parent.mkdir(parents=True, exist_ok=True)
            args.packet_output.write_text(packet, encoding="utf-8")
            _write_yaml(args.decisions_output, decisions)
            result = {
                "status": decisions["status"],
                "packet": str(args.packet_output),
                "decisions": str(args.decisions_output),
            }
        else:
            gold = validate_evidence_review(
                decisions_path=args.decisions,
                chunks_path=args.chunks,
                candidates_path=args.candidates,
            )
            _write_json(args.output, gold)
            result = {"status": gold["status"], "output": str(args.output)}
    except EvidenceReviewError as exc:
        print(json.dumps({"status": "invalid_review", "error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
