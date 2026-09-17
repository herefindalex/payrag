from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from payrag.evidence_review import (
    ATTESTATION,
    EvidenceReviewError,
    prepare_evidence_review,
    validate_evidence_review,
)


class EvidenceReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pilot_path = self.root / "pilot.yaml"
        self.chunks_path = self.root / "chunks.jsonl"
        self.candidates_path = self.root / "candidates.json"
        self.decisions_path = self.root / "decisions.yaml"
        self.pilot_path.write_text(
            yaml.safe_dump(
                {
                    "cases": [
                        {
                            "case_id": "P01",
                            "prompt_en": "How should I retry?",
                            "prompt_zh_TW": "如何重試？",
                        }
                    ]
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        self.chunks_path.write_text(
            json.dumps(
                {
                    "chunk_id": "chunk-1",
                    "source_id": "D01",
                    "section_path": ["Retries"],
                    "text": "Reuse the idempotency key.",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.candidates_path.write_text(
            json.dumps(
                {
                    "snapshot_id": "snapshot",
                    "status": "pending_human_review",
                    "cases": [
                        {
                            "case_id": "P01",
                            "bindings": [
                                {
                                    "evidence_ordinal": 0,
                                    "source_id": "D01",
                                    "role": "required",
                                    "section_or_locator": "Retries",
                                    "candidates": [
                                        {
                                            "rank": 1,
                                            "score": 1.0,
                                            "chunk_id": "chunk-1",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_prepare_is_pending_and_contains_full_text(self) -> None:
        packet, decisions = prepare_evidence_review(
            pilot_path=self.pilot_path,
            chunks_path=self.chunks_path,
            candidates_path=self.candidates_path,
        )

        self.assertEqual("pending_human_review", decisions["status"])
        self.assertEqual(
            [], decisions["cases"][0]["bindings"][0]["accepted_chunk_ids"]
        )
        self.assertIn("Reuse the idempotency key.", packet)

    def test_validate_requires_explicit_complete_review(self) -> None:
        _, decisions = prepare_evidence_review(
            pilot_path=self.pilot_path,
            chunks_path=self.chunks_path,
            candidates_path=self.candidates_path,
        )
        self.decisions_path.write_text(
            yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
        )
        with self.assertRaises(EvidenceReviewError):
            validate_evidence_review(
                decisions_path=self.decisions_path,
                chunks_path=self.chunks_path,
                candidates_path=self.candidates_path,
            )

        decisions.update(
            {
                "status": "approved",
                "reviewer": "human-reviewer",
                "reviewed_at": "2026-09-17T12:00:00Z",
                "attestation": ATTESTATION,
            }
        )
        binding = decisions["cases"][0]["bindings"][0]
        binding["accepted_chunk_ids"] = ["chunk-1"]
        binding["review_status"] = "accepted"
        self.decisions_path.write_text(
            yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
        )

        gold = validate_evidence_review(
            decisions_path=self.decisions_path,
            chunks_path=self.chunks_path,
            candidates_path=self.candidates_path,
        )

        self.assertEqual("approved_human_review", gold["status"])
        self.assertEqual(1, gold["summary"]["binding_count"])
        self.assertEqual(
            "chunk-1", gold["bindings"][0]["accepted_chunks"][0]["chunk_id"]
        )


if __name__ == "__main__":
    unittest.main()
