from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SECRET_PATTERN = re.compile(
    r"\b(?:sk|rk)_(?:test|live)_[A-Za-z0-9]+|\bpi_[A-Za-z0-9]+_secret_[A-Za-z0-9]+"
)


def validate_normalization(snapshot_dir: Path, normalized_dir: Path) -> dict[str, Any]:
    snapshot = json.loads(
        (snapshot_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    blocks = _read_jsonl(normalized_dir / "blocks.jsonl")
    chunks = _read_jsonl(normalized_dir / "chunks.jsonl")
    documents = _read_jsonl(normalized_dir / "documents.jsonl")
    normalization_report = json.loads(
        (normalized_dir / "normalization-report.json").read_text(encoding="utf-8")
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    acquired_results = [r for r in snapshot["results"] if r["status"] == "acquired"]
    hash_failures: list[str] = []
    for result in acquired_results:
        path = snapshot_dir / result["raw_file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != result["sha256"]:
            hash_failures.append(result["source_id"])
    check("raw_snapshot_hashes_match", not hash_failures, hash_failures)

    parsed_sources = {
        document["source_id"]
        for document in documents
        if document.get("status") == "parsed"
    }
    expected_sources = {f"D{number:02d}" for number in range(1, 13)}
    check(
        "all_content_sources_parsed",
        parsed_sources == expected_sources,
        {
            "missing": sorted(expected_sources - parsed_sources),
            "unexpected": sorted(parsed_sources - expected_sources),
        },
    )

    indexed_source_ids = {block["source_id"] for block in blocks} | {
        chunk["source_id"] for chunk in chunks
    }
    check(
        "discovery_index_excluded",
        "D00" not in indexed_source_ids,
        sorted(indexed_source_ids),
    )

    block_ids = [block["block_id"] for block in blocks]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    check("block_ids_unique", len(block_ids) == len(set(block_ids)), len(block_ids))
    check("chunk_ids_unique", len(chunk_ids) == len(set(chunk_ids)), len(chunk_ids))

    known_block_ids = set(block_ids)
    dangling = [
        {"chunk_id": chunk["chunk_id"], "block_id": block_id}
        for chunk in chunks
        for block_id in chunk["block_ids"]
        if block_id not in known_block_ids
    ]
    check("chunk_block_links_resolve", not dangling, dangling[:20])

    max_chunk_characters = normalization_report["max_chunk_characters"]
    oversized = [
        {"chunk_id": chunk["chunk_id"], "characters": chunk["character_count"]}
        for chunk in chunks
        if chunk["character_count"] > max_chunk_characters
    ]
    check(
        "chunk_character_limit",
        not oversized,
        {"configured_limit": max_chunk_characters, "oversized": oversized[:20]},
    )

    combined_text = "\n".join(block["text"] for block in blocks)
    secret_matches = sorted(set(SECRET_PATTERN.findall(combined_text)))
    check(
        "no_secret_shaped_values_in_normalized_text",
        not secret_matches,
        {"match_count": len(secret_matches)},
    )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_source[block["source_id"]].append(block)

    nested_requirements = {
        "D07": ("destination_details", 2),
        "D09": ("data", 2),
        "D10": ("next_action", 2),
    }
    nested_evidence: dict[str, Any] = {}
    nested_ok = True
    for source_id, (prefix, minimum) in nested_requirements.items():
        paths = sorted(
            {
                block.get("metadata", {}).get("field_path")
                for block in by_source[source_id]
                if block["kind"] == "field"
                and (
                    block.get("metadata", {}).get("field_path") == prefix
                    or str(block.get("metadata", {}).get("field_path", "")).startswith(
                        prefix + "."
                    )
                )
            }
        )
        nested_evidence[source_id] = {"prefix": prefix, "path_count": len(paths)}
        nested_ok = nested_ok and len(paths) >= minimum
    check("nested_api_field_paths_preserved", nested_ok, nested_evidence)

    kinds_by_source = {
        source_id: dict(Counter(block["kind"] for block in source_blocks))
        for source_id, source_blocks in by_source.items()
    }
    table_sources = {source_id for source_id, counts in kinds_by_source.items() if counts.get("table")}
    expected_table_sources = {"D02", "D03", "D04", "D05", "D06"}
    check(
        "representative_tables_preserved",
        expected_table_sources <= table_sources,
        {"table_sources": sorted(table_sources)},
    )

    language_blocks = [
        block
        for block in blocks
        if block["kind"] == "code" and block.get("metadata", {}).get("language")
    ]
    check(
        "code_languages_preserved",
        bool(language_blocks),
        {
            "count": len(language_blocks),
            "languages": sorted(
                {block["metadata"]["language"] for block in language_blocks}
            ),
        },
    )

    conditional_blocks = [
        block for block in by_source["D08"] if block.get("metadata", {}).get("conditions")
    ]
    check(
        "version_conditions_preserved",
        bool(conditional_blocks),
        {"D08_conditional_block_count": len(conditional_blocks)},
    )

    scope_terms = {
        "D03": ("duplicate", "order"),
        "D06": ("refund", "uncaptured"),
        "D12": ("paymentintent", "setupintent"),
    }
    scope_evidence: dict[str, dict[str, int]] = {}
    scope_ok = True
    for source_id, terms in scope_terms.items():
        text = "\n".join(block["text"] for block in by_source[source_id]).lower()
        counts = {term: text.count(term) for term in terms}
        scope_evidence[source_id] = counts
        scope_ok = scope_ok and all(count > 0 for count in counts.values())
    check("mixed_scope_markers_preserved", scope_ok, scope_evidence)

    passed = all(item["passed"] for item in checks)
    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "passed": passed,
        "check_count": len(checks),
        "passed_count": sum(item["passed"] for item in checks),
        "failed_count": sum(not item["passed"] for item in checks),
        "checks": checks,
    }
    (normalized_dir / "validation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
