from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from payrag.acquire import fetch_source
from payrag.normalize import extract_article_tree, parse_html_document


NAMED_PROBES: dict[str, tuple[str, ...]] = {
    "D07": ("destination_details",),
    "D09": ("data",),
    "D10": ("next_action",),
}


def with_query_control(url: str, value: str) -> str:
    parts = urlsplit(url)
    query = f"query={quote(value, safe='')}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def probe_expansions(
    manifest: dict[str, Any],
    snapshot_dir: Path,
    *,
    timeout_seconds: float = 30.0,
    attempts: int = 2,
) -> dict[str, Any]:
    snapshot = json.loads(
        (snapshot_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    canonical = {result["source_id"]: result for result in snapshot["results"]}
    variants_dir = snapshot_dir / "query-variants"
    variants_dir.mkdir(exist_ok=True)
    probes: list[dict[str, Any]] = []

    pending_sources = [
        source
        for source in manifest["sources"]
        if source["schema_expansion_status"] == "pending"
    ]
    for source in pending_sources:
        source_id = source["source_id"]
        base_url = source["canonical_url"]
        values = ("", *NAMED_PROBES.get(source_id, ()))
        for value in values:
            label = value or "empty"
            result = fetch_source(
                f"{source_id}__{label}",
                with_query_control(base_url, value),
                variants_dir,
                timeout_seconds=timeout_seconds,
                attempts=attempts,
            )
            canonical_result = canonical.get(source_id, {})
            record = asdict(result)
            record.update(
                {
                    "canonical_source_id": source_id,
                    "query_value": value,
                    "canonical_sha256": canonical_result.get("sha256"),
                    "same_bytes_as_canonical": (
                        result.sha256 is not None
                        and result.sha256 == canonical_result.get("sha256")
                    ),
                    "canonical_byte_count": canonical_result.get("byte_count"),
                    "byte_count_delta": (
                        result.byte_count - canonical_result["byte_count"]
                        if result.byte_count is not None
                        and canonical_result.get("byte_count") is not None
                        else None
                    ),
                }
            )
            if result.status == "acquired" and canonical_result.get("raw_file"):
                canonical_path = snapshot_dir / canonical_result["raw_file"]
                variant_path = snapshot_dir / result.raw_file
                canonical_text = canonical_path.read_text(encoding="utf-8")
                variant_text = variant_path.read_text(encoding="utf-8")
                canonical_tree = extract_article_tree(canonical_text)
                variant_tree = extract_article_tree(variant_text)
                canonical_tree_hash = _article_hash(canonical_tree)
                variant_tree_hash = _article_hash(variant_tree)
                canonical_blocks = parse_html_document(
                    source_id, canonical_result["sha256"], canonical_text
                )
                variant_blocks = parse_html_document(
                    source_id, result.sha256 or "", variant_text
                )
                record.update(
                    {
                        "canonical_article_sha256": canonical_tree_hash,
                        "variant_article_sha256": variant_tree_hash,
                        "same_article_tree_as_canonical": (
                            canonical_tree_hash == variant_tree_hash
                        ),
                        "canonical_block_count": len(canonical_blocks),
                        "variant_block_count": len(variant_blocks),
                        "canonical_field_count": sum(
                            block.kind == "field" for block in canonical_blocks
                        ),
                        "variant_field_count": sum(
                            block.kind == "field" for block in variant_blocks
                        ),
                        "matching_field_paths": sorted(
                            {
                                block.metadata.get("field_path")
                                for block in variant_blocks
                                if value
                                and block.kind == "field"
                                and (
                                    block.metadata.get("field_path") == value
                                    or str(block.metadata.get("field_path", "")).startswith(
                                        value + "."
                                    )
                                )
                            }
                        ),
                    }
                )
            probes.append(record)

    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "probe_count": len(probes),
        "acquired_count": sum(probe["status"] == "acquired" for probe in probes),
        "failed_count": sum(probe["status"] == "failed" for probe in probes),
        "all_article_trees_identical_to_canonical": all(
            probe.get("same_article_tree_as_canonical") is True for probe in probes
        ),
        "interpretation": (
            "Hash and size differences only establish that the documentation response changed. "
            "Field completeness still requires parser-level comparison."
        ),
        "probes": probes,
    }
    (snapshot_dir / "query-expansion-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _article_hash(tree: dict[str, Any]) -> str:
    encoded = json.dumps(
        tree, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
