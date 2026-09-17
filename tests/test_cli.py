from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from payrag.cli import main


class _EmptyIndex:
    def search(self, query: str, *, top_k: int, max_per_source=None):
        return []


class CliTests(unittest.TestCase):
    def test_prepare_context_loads_selected_dense_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "context.json"
            with patch("payrag.cli.DenseIndex.load", return_value=_EmptyIndex()) as load:
                status = main(
                    [
                        "prepare-context",
                        "dense.json",
                        "question",
                        "--retriever",
                        "dense",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(0, status)
            load.assert_called_once()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("dense", payload["retrieval"]["kind"])

    def test_search_bm25_does_not_require_retriever_option(self) -> None:
        with patch("payrag.cli.BM25Index.load", return_value=_EmptyIndex()):
            status = main(["search-bm25", "bm25.json", "question"])
        self.assertEqual(0, status)


if __name__ == "__main__":
    unittest.main()
