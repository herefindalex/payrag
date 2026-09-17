from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    return TOKEN_PATTERN.findall(expanded.lower())


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    source_id: str
    chunk_id: str
    section_path: list[str]
    text: str
    parent_chunk_id: str | None = None


class BM25Index:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 index without chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for chunk in chunks:
            section = " ".join(chunk.get("section_path", []))
            tokens = tokenize(f"{section} {section} {chunk['text']}")
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())

        self.document_frequency = document_frequency
        self.average_document_length = sum(self.document_lengths) / len(
            self.document_lengths
        )

    @classmethod
    def from_chunks_jsonl(cls, path: Path) -> "BM25Index":
        chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return cls(chunks)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["chunks"], k1=payload["k1"], b=payload["b"])

    def save(self, path: Path, *, snapshot_id: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "bm25",
            "snapshot_id": snapshot_id,
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(self.chunks),
            "average_document_length": self.average_document_length,
            "tokenizer": "lowercase alphanumeric with snake_case and camelCase splitting",
            "chunks": self.chunks,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_per_source: int | None = None,
    ) -> list[SearchResult]:
        query_terms = Counter(tokenize(query))
        if not query_terms or top_k < 1:
            return []
        document_count = len(self.chunks)
        scored: list[tuple[float, int]] = []

        for index, frequencies in enumerate(self.term_frequencies):
            length = self.document_lengths[index]
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_document_length
                )
                score += query_frequency * idf * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]]["chunk_id"]))
        selected: list[tuple[float, int]] = []
        source_counts: Counter[str] = Counter()
        for score, index in scored:
            source_id = self.chunks[index]["source_id"]
            if max_per_source is not None and source_counts[source_id] >= max_per_source:
                continue
            selected.append((score, index))
            source_counts[source_id] += 1
            if len(selected) >= top_k:
                break

        results: list[SearchResult] = []
        for rank, (score, index) in enumerate(selected, start=1):
            chunk = self.chunks[index]
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(score, 6),
                    source_id=chunk["source_id"],
                    chunk_id=chunk["chunk_id"],
                    section_path=chunk.get("section_path", []),
                    text=chunk["text"],
                    parent_chunk_id=chunk.get("parent_chunk_id"),
                )
            )
        return results


def result_record(result: SearchResult, *, include_text: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rank": result.rank,
        "score": result.score,
        "source_id": result.source_id,
        "chunk_id": result.chunk_id,
        "section_path": result.section_path,
        "excerpt": re.sub(r"\s+", " ", result.text)[:320],
    }
    if include_text:
        record["text"] = result.text
    if result.parent_chunk_id:
        record["parent_chunk_id"] = result.parent_chunk_id
    return record
