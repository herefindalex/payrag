from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ManifestError(ValueError):
    """Raised when a source manifest violates the local contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KINDS = {"discovery_only", "api_reference", "guide"}
_RAW_STATUSES = {"not_acquired", "acquired", "failed"}
_PARSER_STATUSES = {"not_run", "passed", "failed", "partial"}
_EXPANSION_STATUSES = {"not_applicable", "pending", "passed", "failed", "partial"}


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Unable to read manifest {path}: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = {"project", "version", "created_on", "source_count", "sources"}
    missing = required - manifest.keys()
    if missing:
        raise ManifestError(f"Missing top-level fields: {', '.join(sorted(missing))}")

    sources = manifest["sources"]
    if not isinstance(sources, list):
        raise ManifestError("sources must be a list")
    if manifest["source_count"] != len(sources):
        raise ManifestError(
            f"source_count={manifest['source_count']} does not match {len(sources)} sources"
        )

    seen_ids: set[str] = set()
    for position, source in enumerate(sources):
        _validate_source(source, position, seen_ids)


def _validate_source(source: Any, position: int, seen_ids: set[str]) -> None:
    if not isinstance(source, dict):
        raise ManifestError(f"sources[{position}] must be an object")

    required = {
        "source_id",
        "title",
        "canonical_url",
        "source_kind",
        "raw_snapshot_status",
        "raw_sha256",
        "parser_status",
        "schema_expansion_status",
        "index_eligibility",
    }
    missing = required - source.keys()
    if missing:
        raise ManifestError(
            f"sources[{position}] missing fields: {', '.join(sorted(missing))}"
        )

    source_id = source["source_id"]
    if not isinstance(source_id, str) or not re.fullmatch(r"D\d{2}", source_id):
        raise ManifestError(f"sources[{position}].source_id must match DNN")
    if source_id in seen_ids:
        raise ManifestError(f"Duplicate source_id: {source_id}")
    seen_ids.add(source_id)

    parsed_url = urlparse(source["canonical_url"])
    if parsed_url.scheme != "https" or parsed_url.netloc != "docs.stripe.com":
        raise ManifestError(f"{source_id} must use an https://docs.stripe.com URL")

    _require_choice(source_id, "source_kind", source["source_kind"], _SOURCE_KINDS)
    _require_choice(
        source_id, "raw_snapshot_status", source["raw_snapshot_status"], _RAW_STATUSES
    )
    _require_choice(source_id, "parser_status", source["parser_status"], _PARSER_STATUSES)
    _require_choice(
        source_id,
        "schema_expansion_status",
        source["schema_expansion_status"],
        _EXPANSION_STATUSES,
    )

    raw_status = source["raw_snapshot_status"]
    raw_sha256 = source["raw_sha256"]
    if raw_status == "acquired":
        if not isinstance(raw_sha256, str) or not _SHA256_RE.fullmatch(raw_sha256):
            raise ManifestError(f"{source_id} acquired snapshots require a SHA-256")
    elif raw_sha256 is not None:
        raise ManifestError(f"{source_id} must not have a hash before acquisition")


def _require_choice(source_id: str, field: str, value: Any, choices: set[str]) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ManifestError(f"{source_id}.{field} must be one of: {allowed}")
