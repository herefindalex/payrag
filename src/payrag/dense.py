from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from payrag.retrieval import SearchResult


class DenseIndex:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: Any,
        model: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        chunks_path: Path,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> "DenseIndex":
        import numpy as np
        import sentence_transformers
        import torch
        from sentence_transformers import SentenceTransformer

        chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
        ]
        if not chunks:
            raise ValueError("Cannot build a dense index without chunks")
        model = SentenceTransformer(model_name, device=device, revision=revision)
        auto_model = getattr(model[0], "auto_model", None)
        resolved_revision = getattr(
            getattr(auto_model, "config", None), "_commit_hash", None
        )
        input_texts = [_document_text(chunk) for chunk in chunks]
        input_token_lengths = [
            len(
                model.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in input_texts
        ]
        max_sequence_length = int(model.max_seq_length)
        dense_chunks, split_input_chunk_count = _split_chunks_for_model(
            chunks,
            model.tokenizer,
            max_sequence_length=max_sequence_length,
            model_name=model_name,
        )
        texts = [_document_text(chunk) for chunk in dense_chunks]
        token_lengths = [
            len(
                model.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in texts
        ]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        metadata = {
            "kind": "dense_cosine",
            "model_name": model_name,
            "model_revision": resolved_revision or revision,
            "device": device,
            "embedding_dimension": int(embeddings.shape[1]),
            "input_chunk_count": len(chunks),
            "chunk_count": len(dense_chunks),
            "max_sequence_length": max_sequence_length,
            "pre_split_token_length_distribution": _distribution(
                input_token_lengths
            ),
            "token_length_distribution": _distribution(token_lengths),
            "split_input_chunk_count": split_input_chunk_count,
            "truncated_chunk_count": sum(
                length > max_sequence_length for length in token_lengths
            ),
            "package_versions": {
                "numpy": np.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "torch": torch.__version__,
            },
            "chunks": dense_chunks,
        }
        return cls(dense_chunks, embeddings, model, metadata)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "DenseIndex":
        import numpy as np
        from sentence_transformers import SentenceTransformer

        metadata = json.loads(path.read_text(encoding="utf-8"))
        embeddings_path = path.with_suffix(".npz")
        embeddings = np.load(embeddings_path)["embeddings"]
        model = SentenceTransformer(
            metadata["model_name"],
            device=device,
            revision=metadata.get("model_revision"),
        )
        return cls(metadata["chunks"], embeddings, model, metadata)

    def save(self, path: Path, *, snapshot_id: str | None = None) -> None:
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "snapshot_id": snapshot_id}
        path.write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(path.with_suffix(".npz"), embeddings=self.embeddings)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_per_source: int | None = None,
    ) -> list[SearchResult]:
        import numpy as np

        if not query.strip() or top_k < 1:
            return []
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        ordered = np.argsort(-scores, kind="stable")
        selected: list[int] = []
        source_counts: dict[str, int] = {}
        for index_value in ordered:
            index = int(index_value)
            source_id = self.chunks[index]["source_id"]
            if (
                max_per_source is not None
                and source_counts.get(source_id, 0) >= max_per_source
            ):
                continue
            selected.append(index)
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            if len(selected) >= top_k:
                break

        return [
            SearchResult(
                rank=rank,
                score=round(float(scores[index]), 6),
                source_id=self.chunks[index]["source_id"],
                chunk_id=self.chunks[index]["chunk_id"],
                section_path=self.chunks[index].get("section_path", []),
                text=self.chunks[index]["text"],
                parent_chunk_id=self.chunks[index].get("parent_chunk_id"),
            )
            for rank, index in enumerate(selected, start=1)
        ]


def _document_text(chunk: dict[str, Any]) -> str:
    section = " > ".join(chunk.get("section_path", []))
    return f"{section}\n{chunk['text']}" if section else chunk["text"]


def _split_chunks_for_model(
    chunks: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    model_name: str,
    overlap_tokens: int = 32,
) -> tuple[list[dict[str, Any]], int]:
    dense_chunks: list[dict[str, Any]] = []
    split_input_chunk_count = 0
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)

    for chunk in chunks:
        document_tokens = tokenizer.encode(
            _document_text(chunk), add_special_tokens=True, truncation=False
        )
        if len(document_tokens) <= max_sequence_length:
            dense_chunks.append(chunk)
            continue

        split_input_chunk_count += 1
        section = " > ".join(chunk.get("section_path", []))
        prefix = f"{section}\n" if section else ""
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        available = max_sequence_length - special_token_count - len(prefix_tokens)
        if available < 64:
            prefix = ""
            prefix_tokens = []
            available = max_sequence_length - special_token_count
        step = max(1, available - min(overlap_tokens, available // 4))
        body_tokens = tokenizer.encode(
            chunk["text"], add_special_tokens=False, truncation=False
        )
        windows = [
            body_tokens[start : start + available]
            for start in range(0, len(body_tokens), step)
        ]
        if len(windows) > 1 and len(windows[-1]) <= overlap_tokens:
            windows.pop()

        for index, window in enumerate(windows, start=1):
            text = tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            derived_id = hashlib.sha256(
                f"{chunk['chunk_id']}\0{model_name}\0{index}\0{text}".encode("utf-8")
            ).hexdigest()[:24]
            dense_chunks.append(
                {
                    **chunk,
                    "chunk_id": derived_id,
                    "parent_chunk_id": chunk["chunk_id"],
                    "dense_segment_index": index,
                    "dense_segment_total": len(windows),
                    "text": text,
                    "character_count": len(text),
                    "utf8_byte_count": len(text.encode("utf-8")),
                }
            )

    return dense_chunks, split_input_chunk_count


def _distribution(values: list[int]) -> dict[str, int | float]:
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
