from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


INITIAL_STATE_MARKER = "window.__INITIAL_STATE__ = "
_KEY_PATTERN = re.compile(
    r"\b(?:sk|rk)_(?:test|live)_[A-Za-z0-9]+"
    r"|\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}"
    r"|\bwhsec_[A-Za-z0-9]+"
    r"|\b(?:pi|seti)_[A-Za-z0-9]+_secret_[A-Za-z0-9]+"
)
_LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


@dataclass(frozen=True)
class Block:
    source_id: str
    block_id: str
    ordinal: int
    kind: str
    section_path: list[str]
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_id: str
    ordinal: int
    section_path: list[str]
    parent_section_id: str
    block_ids: list[str]
    text: str
    character_count: int
    utf8_byte_count: int
    whitespace_token_count: int


def extract_article_tree(html: str) -> dict[str, Any]:
    marker_index = html.find(INITIAL_STATE_MARKER)
    if marker_index < 0:
        raise ValueError("Stripe initial state marker is missing")
    json_start = marker_index + len(INITIAL_STATE_MARKER)
    state, _ = json.JSONDecoder().raw_decode(html[json_start:])
    try:
        article = state["article"]["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Stripe initial state has no article content") from exc
    if not isinstance(article, dict):
        raise ValueError("Stripe article content is not a document node")
    return article


def parse_html_document(source_id: str, source_sha256: str, html: str) -> list[Block]:
    renderer = _Renderer(source_id=source_id, source_sha256=source_sha256)
    renderer.visit(extract_article_tree(html))
    return renderer.blocks


def parse_markdown_document(
    source_id: str, source_sha256: str, markdown: str
) -> list[Block]:
    blocks: list[Block] = []
    section_path: list[str] = []
    buffer: list[str] = []

    def flush(kind: str = "paragraph") -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        buffer = []
        if text:
            blocks.append(
                _make_block(
                    source_id,
                    source_sha256,
                    len(blocks),
                    kind,
                    section_path,
                    text,
                    {},
                )
            )

    in_fence = False
    fence_language = ""
    for line in markdown.splitlines():
        if line.startswith("```"):
            if in_fence:
                flush("code")
                in_fence = False
                fence_language = ""
            else:
                flush()
                in_fence = True
                fence_language = line[3:].strip()
            continue
        if in_fence:
            buffer.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2)
            section_path[:] = section_path[: level - 1]
            section_path.append(title)
            blocks.append(
                _make_block(
                    source_id,
                    source_sha256,
                    len(blocks),
                    "heading",
                    section_path,
                    title,
                    {"level": level},
                )
            )
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush("code" if in_fence else "paragraph")
    return blocks


def build_chunks(
    source_id: str,
    source_sha256: str,
    blocks: list[Block],
    *,
    max_characters: int = 6000,
) -> list[Chunk]:
    if max_characters < 500:
        raise ValueError("max_characters must be at least 500")

    groups: list[list[Block]] = []
    current: list[Block] = []
    current_size = 0
    current_section: tuple[str, ...] | None = None

    for block in blocks:
        pieces = _split_oversized_block(block, max_characters)
        for piece in pieces:
            section = tuple(piece.section_path)
            size = len(piece.text) + 2
            if current and (section != current_section or current_size + size > max_characters):
                groups.append(current)
                current = []
                current_size = 0
            current.append(piece)
            current_size += size
            current_section = section
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for ordinal, group in enumerate(groups):
        text = "\n\n".join(block.text for block in group).strip()
        section_path = group[0].section_path
        parent_section_id = _stable_hash(
            source_id, source_sha256, "section", *section_path
        )[:24]
        chunk_id = _stable_hash(
            source_id,
            source_sha256,
            "chunk",
            str(ordinal),
            *section_path,
            text,
        )[:24]
        chunks.append(
            Chunk(
                source_id=source_id,
                chunk_id=chunk_id,
                ordinal=ordinal,
                section_path=list(section_path),
                parent_section_id=parent_section_id,
                block_ids=[block.block_id for block in group],
                text=text,
                character_count=len(text),
                utf8_byte_count=len(text.encode("utf-8")),
                whitespace_token_count=len(re.findall(r"\S+", text)),
            )
        )
    return chunks


def normalize_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    max_characters: int = 6000,
) -> dict[str, Any]:
    snapshot = json.loads(
        (snapshot_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    blocks_path = output_dir / "blocks.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    document_records: list[dict[str, Any]] = []
    all_blocks: list[Block] = []
    all_chunks: list[Chunk] = []
    failures: list[dict[str, str]] = []

    for result in snapshot["results"]:
        source_id = result["source_id"]
        if result["status"] != "acquired" or not result["raw_file"]:
            failures.append({"source_id": source_id, "error": "source not acquired"})
            continue
        if source_id == "D00":
            document_records.append(
                {
                    "source_id": source_id,
                    "status": "discovery_only",
                    "indexed": False,
                    "raw_sha256": result["sha256"],
                }
            )
            continue

        raw_path = snapshot_dir / result["raw_file"]
        raw_text = raw_path.read_text(encoding="utf-8")
        try:
            if raw_path.suffix == ".html":
                blocks = parse_html_document(source_id, result["sha256"], raw_text)
            else:
                blocks = parse_markdown_document(source_id, result["sha256"], raw_text)
            chunks = build_chunks(
                source_id,
                result["sha256"],
                blocks,
                max_characters=max_characters,
            )
        except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            failures.append({"source_id": source_id, "error": str(exc)})
            continue

        all_blocks.extend(blocks)
        all_chunks.extend(chunks)
        document_records.append(
            {
                "source_id": source_id,
                "status": "parsed",
                "indexed": True,
                "raw_sha256": result["sha256"],
                "block_count": len(blocks),
                "chunk_count": len(chunks),
            }
        )

    _write_jsonl(documents_path, document_records)
    _write_jsonl(blocks_path, (asdict(block) for block in all_blocks))
    _write_jsonl(chunks_path, (asdict(chunk) for chunk in all_chunks))

    sizes = [chunk.character_count for chunk in all_chunks]
    whitespace_tokens = [chunk.whitespace_token_count for chunk in all_chunks]
    report = {
        "snapshot_id": snapshot["snapshot_id"],
        "source_count": len(snapshot["results"]),
        "parsed_document_count": sum(r.get("status") == "parsed" for r in document_records),
        "discovery_only_count": sum(
            r.get("status") == "discovery_only" for r in document_records
        ),
        "failure_count": len(failures),
        "failures": failures,
        "block_count": len(all_blocks),
        "chunk_count": len(all_chunks),
        "max_chunk_characters": max_characters,
        "chunk_character_distribution": _distribution(sizes),
        "chunk_whitespace_token_distribution": _distribution(whitespace_tokens),
        "token_measurement_note": (
            "Whitespace token counts are parser diagnostics, not model tokenizer counts. "
            "Model-token distributions must be measured after the embedding and generation "
            "models are selected."
        ),
    }
    (output_dir / "normalization-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


class _Renderer:
    def __init__(self, source_id: str, source_sha256: str) -> None:
        self.source_id = source_id
        self.source_sha256 = source_sha256
        self.blocks: list[Block] = []
        self.section_path: list[str] = []
        self.conditions: list[dict[str, Any]] = []
        self.field_path: list[str] = []
        self.code_context: list[dict[str, Any]] = []

    def visit(self, node: Any) -> None:
        if isinstance(node, str):
            return
        if isinstance(node, list):
            for child in node:
                self.visit(child)
            return
        if not isinstance(node, dict):
            return

        name = node.get("name") or node.get("$$mdtype") or "unknown"
        attributes = node.get("attributes", {})
        children = node.get("children", [])

        if name == "ApiSection":
            title = str(attributes.get("title") or attributes.get("route") or "API section")
            self.section_path = [title]
            self.add("heading", title, {"level": 1, **_scalar_metadata(attributes)})
            for child in children:
                self.visit(child)
            return
        if name == "Heading":
            title = _inline_text(children).strip()
            level = int(attributes.get("level", 2))
            self.section_path = self.section_path[: max(0, level - 1)]
            self.section_path.append(title)
            self.add("heading", title, {"level": level, **_scalar_metadata(attributes)})
            return
        if name == "Paragraph":
            self.add("paragraph", _inline_text(children), {})
            return
        if name == "Callout":
            self.add(
                "callout",
                _inline_text(children),
                {"callout_type": attributes.get("type")},
            )
            return
        if name == "Table":
            rows = _table_rows(node)
            if rows:
                self.add(
                    "table",
                    "\n".join(" | ".join(cell for cell in row) for row in rows),
                    {"rows": len(rows), **_scalar_metadata(attributes)},
                )
            return
        if name == "List":
            ordered = bool(attributes.get("ordered"))
            items = [
                _inline_text(child.get("children", [])).strip()
                for child in children
                if isinstance(child, dict)
            ]
            text = "\n".join(
                f"{index + 1 if ordered else '-'}{'.' if ordered else ''} {item}"
                for index, item in enumerate(items)
                if item
            )
            self.add("list", text, {"ordered": ordered})
            return
        if name == "CodeTabGroup":
            items = attributes.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict) or not isinstance(item.get("content"), dict):
                        continue
                    self.code_context.append(
                        {
                            "code_tab_id": item.get("id"),
                            "code_tab_title": item.get("title"),
                        }
                    )
                    self.visit(item["content"])
                    self.code_context.pop()
            else:
                for child in children:
                    self.visit(child)
            return
        if name in {"CodeBlock", "Fence"}:
            plaintext = attributes.get("plaintext")
            text = plaintext if isinstance(plaintext, str) else _inline_text(children)
            code_metadata = {
                "language": attributes.get("language"),
                "title": attributes.get("title"),
            }
            if self.code_context:
                code_metadata.update(self.code_context[-1])
            self.add(
                "code",
                text,
                code_metadata,
            )
            return
        if name == "ElementList":
            title = attributes.get("title")
            previous_path = list(self.section_path)
            if title:
                self.section_path.append(str(title))
            for child in children:
                self.visit(child)
            self.section_path = previous_path
            return
        if name == "Element":
            field_name = str(attributes.get("name") or "unnamed")
            self.field_path.append(field_name)
            description_children = [
                child
                for child in children
                if not (
                    isinstance(child, dict)
                    and (child.get("name") or child.get("$$mdtype"))
                    in {"Element", "ElementList", "EnumValuesList"}
                )
            ]
            description = _inline_text(description_children).strip()
            qualified_name = ".".join(self.field_path)
            prefix = qualified_name
            field_type = attributes.get("type")
            if field_type:
                prefix += f" ({field_type})"
            self.add(
                "field",
                f"{prefix}: {description}" if description else prefix,
                {**_scalar_metadata(attributes), "field_path": qualified_name},
            )
            for child in children:
                if isinstance(child, dict) and (
                    child.get("name") or child.get("$$mdtype")
                ) in {"Element", "ElementList", "EnumValuesList"}:
                    self.visit(child)
            self.field_path.pop()
            return
        if name == "EnumValuesList":
            values = [
                str(child.get("attributes", {}).get("name"))
                for child in children
                if isinstance(child, dict) and child.get("attributes", {}).get("name")
            ]
            field_path = ".".join(self.field_path)
            label = f"{field_path} allowed values" if field_path else "Allowed values"
            self.add(
                "enum",
                f"{label}: {', '.join(values)}",
                {"total": attributes.get("total"), "field_path": field_path or None},
            )
            return
        if name == "When":
            self.conditions.append(attributes)
            for child in children:
                self.visit(child)
            self.conditions.pop()
            return

        for child in children:
            self.visit(child)

    def add(self, kind: str, text: str, metadata: dict[str, Any]) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return
        if self.conditions:
            metadata = {
                **metadata,
                "conditions": [dict(condition) for condition in self.conditions],
            }
        self.blocks.append(
            _make_block(
                self.source_id,
                self.source_sha256,
                len(self.blocks),
                kind,
                self.section_path,
                normalized,
                metadata,
            )
        )


def _make_block(
    source_id: str,
    source_sha256: str,
    ordinal: int,
    kind: str,
    section_path: list[str],
    text: str,
    metadata: dict[str, Any],
) -> Block:
    block_id = _stable_hash(
        source_id,
        source_sha256,
        "block",
        str(ordinal),
        kind,
        *section_path,
        text,
        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )[:24]
    return Block(
        source_id=source_id,
        block_id=block_id,
        ordinal=ordinal,
        kind=kind,
        section_path=list(section_path),
        text=text,
        metadata=metadata,
    )


def _inline_text(children: Any) -> str:
    if isinstance(children, str):
        return children
    if isinstance(children, list):
        return "".join(_inline_text(child) for child in children)
    if not isinstance(children, dict):
        return ""
    name = children.get("name") or children.get("$$mdtype")
    attributes = children.get("attributes", {})
    if name == "InlineCode":
        return f"`{attributes.get('content', '')}`"
    if name == "ApiKey":
        return "<redacted-api-key>"
    if name == "Link":
        label = _inline_text(children.get("children", []))
        href = attributes.get("href")
        return f"[{label}]({href})" if href else label
    if name == "CodeLine":
        tokens = attributes.get("tokens")
        if isinstance(tokens, list):
            return "".join(str(token.get("content", "")) for token in tokens if isinstance(token, dict))
    text = _inline_text(children.get("children", []))
    if name == "strong":
        return f"**{text}**"
    if name == "em":
        return f"_{text}_"
    return text


def _table_rows(node: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []

    def walk(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        name = item.get("name") or item.get("$$mdtype")
        if name == "TableRow":
            cells = []
            for child in item.get("children", []):
                if isinstance(child, dict):
                    cell = _normalize_text(_inline_text(child.get("children", [])))
                    cells.append(cell)
            if cells:
                rows.append(cells)
            return
        walk(item.get("children", []))

    walk(node.get("children", []))
    return rows


def _scalar_metadata(attributes: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in attributes.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _normalize_text(text: str) -> str:
    text = redact_sensitive_values(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def redact_sensitive_values(text: str) -> str:
    text = _KEY_PATTERN.sub("<redacted-secret>", text)

    def redact_long_number(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _passes_luhn(digits):
            return "<redacted-pan>"
        return match.group(0)

    return _LONG_NUMBER_PATTERN.sub(redact_long_number, text)


def _passes_luhn(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _split_oversized_block(block: Block, max_characters: int) -> list[Block]:
    if len(block.text) <= max_characters:
        return [block]
    lines = block.text.splitlines() or [block.text]
    pieces: list[str] = []
    current = ""
    for line in lines:
        candidates = [line[i : i + max_characters] for i in range(0, len(line), max_characters)] or [""]
        for candidate in candidates:
            joined = f"{current}\n{candidate}".strip() if current else candidate
            if current and len(joined) > max_characters:
                pieces.append(current)
                current = candidate
            else:
                current = joined
    if current:
        pieces.append(current)
    return [
        Block(
            source_id=block.source_id,
            block_id=block.block_id,
            ordinal=block.ordinal,
            kind=block.kind,
            section_path=block.section_path,
            text=piece,
            metadata={**block.metadata, "split_part": index + 1, "split_total": len(pieces)},
        )
        for index, piece in enumerate(pieces)
    ]


def _stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }
