from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import yaml

from payrag.evaluate import evaluate_development_pilot
from payrag.retrieval import SearchResult


class _Retriever:
    def search(self, query, *, top_k=5, max_per_source=None):
        return [
            SearchResult(
                rank=1,
                score=1.0,
                source_id="D01",
                chunk_id="chunk-1",
                section_path=["Retry"],
                text="Retry safely.",
            )
        ]


class EvaluateTests(unittest.TestCase):
    def test_retrieval_only_case_does_not_require_answer_key(self) -> None:
        pilot = {
            "dataset_version": "public-retrieval-v1",
            "cases": [
                {
                    "case_id": "P01",
                    "family": "retry",
                    "prompt_en": "How should I retry?",
                    "evidence": [{"source_id": "D01"}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pilot_path = root / "pilot.yaml"
            output_path = root / "report.json"
            pilot_path.write_text(yaml.safe_dump(pilot), encoding="utf-8")
            report = evaluate_development_pilot(
                _Retriever(), pilot_path, output_path
            )

        self.assertEqual(report["cases"][0]["case_id"], "P01")
        self.assertIsNone(report["cases"][0]["expected_decision"])


if __name__ == "__main__":
    unittest.main()
