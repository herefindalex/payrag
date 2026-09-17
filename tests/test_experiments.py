from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from payrag.experiments import (
    build_evidence_binding_candidates,
    build_m2_context_manifest,
)
from payrag.retrieval import SearchResult


class _FakeRetriever:
    def __init__(self, result: SearchResult) -> None:
        self.result = result

    def search(self, query, *, top_k=5, max_per_source=None):
        return [self.result]


class ExperimentArtifactTests(unittest.TestCase):
    def test_candidates_are_pending_and_do_not_modify_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pilot_path = root / "pilot.yaml"
            pilot = {
                "dataset_version": "test",
                "cases": [
                    {
                        "case_id": "P01",
                        "evidence": [
                            {
                                "source_id": "D01",
                                "section_or_locator": "Network errors",
                                "role": "required",
                                "snapshot_chunk_ids": [],
                            }
                        ],
                    }
                ],
            }
            pilot_path.write_text(yaml.safe_dump(pilot), encoding="utf-8")
            original = pilot_path.read_bytes()
            chunks_path = root / "chunks.jsonl"
            chunks_path.write_text(
                json.dumps(
                    {
                        "source_id": "D01",
                        "chunk_id": "chunk-1",
                        "section_path": ["Network errors"],
                        "text": "Retry network errors with the same idempotency key.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_evidence_binding_candidates(
                snapshot_id="snapshot",
                pilot_path=pilot_path,
                chunks_path=chunks_path,
            )

            self.assertEqual(report["status"], "pending_human_review")
            binding = report["cases"][0]["bindings"][0]
            self.assertEqual(binding["review_status"], "pending_human_review")
            self.assertEqual(binding["candidates"][0]["chunk_id"], "chunk-1")
            self.assertEqual(pilot_path.read_bytes(), original)

    def test_context_manifest_keeps_oracle_and_ablation_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pilot_path = root / "pilot.yaml"
            pilot_path.write_text(
                yaml.safe_dump(
                    {
                        "cases": [
                            {
                                "case_id": "P01",
                                "prompt_en": "What should I retry?",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            source_manifest_path = root / "sources.json"
            source_manifest_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_id": "D01",
                                "canonical_url": "https://example.test/d01",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidates_path = root / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    {
                        "snapshot_id": "snapshot",
                        "status": "pending_human_review",
                    }
                ),
                encoding="utf-8",
            )
            result = SearchResult(
                rank=1,
                score=1.0,
                source_id="D01",
                chunk_id="chunk-1",
                section_path=["Retry"],
                text="Use the same key.",
            )

            manifest = build_m2_context_manifest(
                snapshot_id="snapshot",
                pilot_path=pilot_path,
                source_manifest_path=source_manifest_path,
                bm25=_FakeRetriever(result),
                dense=_FakeRetriever(result),
                evidence_candidates_path=candidates_path,
            )

            variants = manifest["cases"][0]["variants"]
            self.assertEqual(variants["no_context"]["evidence"], [])
            self.assertEqual(variants["bm25"]["evidence"][0]["chunk_id"], "chunk-1")
            self.assertEqual(variants["dense"]["evidence"][0]["chunk_id"], "chunk-1")
            self.assertEqual(variants["oracle"]["status"], "blocked_pending_human_review")
            self.assertEqual(
                variants["evidence_ablation"]["status"],
                "blocked_pending_human_review",
            )
            self.assertFalse(manifest["comparison_controls"]["claim_eligible"])


if __name__ == "__main__":
    unittest.main()
