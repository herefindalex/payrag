from __future__ import annotations

import hashlib
import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


USER_AGENT = "payrag-research/0.1 (+private documentation evaluation)"


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    requested_url: str
    final_url: str | None
    retrieved_at: str
    http_status: int | None
    content_type: str | None
    etag: str | None
    last_modified: str | None
    sha256: str | None
    byte_count: int | None
    raw_file: str | None
    status: str
    error_type: str | None
    error_message: str | None
    elapsed_ms: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_id(now: datetime | None = None) -> str:
    timestamp = now or utc_now()
    return timestamp.strftime("%Y%m%dT%H%M%SZ")


def acquire_sources(
    manifest: dict[str, Any],
    output_root: Path,
    *,
    source_ids: set[str] | None = None,
    timeout_seconds: float = 30.0,
    attempts: int = 2,
) -> tuple[Path, list[FetchResult]]:
    run_dir = output_root / snapshot_id()
    raw_dir = run_dir / "responses"
    raw_dir.mkdir(parents=True, exist_ok=False)

    selected = [
        source
        for source in manifest["sources"]
        if source_ids is None or source["source_id"] in source_ids
    ]
    results: list[FetchResult] = []
    for source in selected:
        result = fetch_source(
            source["source_id"],
            source["canonical_url"],
            raw_dir,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
        )
        results.append(result)

    snapshot_manifest = {
        "project": manifest["project"],
        "source_manifest_version": manifest["version"],
        "snapshot_id": run_dir.name,
        "created_at": utc_now().isoformat().replace("+00:00", "Z"),
        "result_count": len(results),
        "results": [asdict(result) for result in results],
    }
    (run_dir / "snapshot-manifest.json").write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, results


def fetch_source(
    source_id: str,
    url: str,
    raw_dir: Path,
    *,
    timeout_seconds: float,
    attempts: int,
) -> FetchResult:
    started = time.monotonic()
    retrieved_at = utc_now().isoformat().replace("+00:00", "Z")
    last_error: BaseException | None = None

    for attempt in range(1, max(1, attempts) + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
                "User-Agent": USER_AGENT,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
                status = response.status
                final_url = response.geturl()
                headers = response.headers
            digest = hashlib.sha256(body).hexdigest()
            suffix = _suffix_for(headers.get_content_type())
            filename = f"{source_id}{suffix}"
            path = raw_dir / filename
            path.write_bytes(body)
            return FetchResult(
                source_id=source_id,
                requested_url=url,
                final_url=final_url,
                retrieved_at=retrieved_at,
                http_status=status,
                content_type=headers.get("Content-Type"),
                etag=headers.get("ETag"),
                last_modified=headers.get("Last-Modified"),
                sha256=digest,
                byte_count=len(body),
                raw_file=str(path.relative_to(raw_dir.parent)),
                status="acquired",
                error_type=None,
                error_message=None,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(attempt, 2))

    return FetchResult(
        source_id=source_id,
        requested_url=url,
        final_url=getattr(last_error, "url", None),
        retrieved_at=retrieved_at,
        http_status=getattr(last_error, "code", None),
        content_type=None,
        etag=None,
        last_modified=None,
        sha256=None,
        byte_count=None,
        raw_file=None,
        status="failed",
        error_type=type(last_error).__name__ if last_error else "UnknownError",
        error_message=str(last_error) if last_error else "Unknown acquisition failure",
        elapsed_ms=round((time.monotonic() - started) * 1000),
    )


def summarize_results(results: Iterable[FetchResult]) -> dict[str, int]:
    summary = {"acquired": 0, "failed": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def _suffix_for(content_type: str) -> str:
    if content_type == "text/html":
        return ".html"
    if content_type in {"text/plain", "text/markdown"}:
        return ".txt"
    if content_type == "application/json":
        return ".json"
    return ".bin"
