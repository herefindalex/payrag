from __future__ import annotations

import unittest

from payrag.retrieval import BM25Index, tokenize


class RetrievalTests(unittest.TestCase):
    def test_tokenizer_splits_snake_case_and_camel_case(self) -> None:
        self.assertEqual(
            ["payment", "intent", "next", "action"],
            tokenize("PaymentIntent.next_action"),
        )

    def test_bm25_ranks_matching_chunk_first(self) -> None:
        chunks = [
            {
                "source_id": "D01",
                "chunk_id": "one",
                "section_path": ["Idempotency"],
                "text": "Reuse the same idempotency key and parameters after a network error.",
            },
            {
                "source_id": "D06",
                "chunk_id": "two",
                "section_path": ["Refunds"],
                "text": "A pending refund has not necessarily reached the customer.",
            },
        ]
        results = BM25Index(chunks).search(
            "network retry idempotency key refund", top_k=2
        )
        self.assertEqual("one", results[0].chunk_id)
        self.assertGreater(results[0].score, results[1].score)

    def test_source_cap_diversifies_results(self) -> None:
        chunks = [
            {"source_id": "D01", "chunk_id": "a", "section_path": [], "text": "retry key"},
            {"source_id": "D01", "chunk_id": "b", "section_path": [], "text": "retry key again"},
            {"source_id": "D02", "chunk_id": "c", "section_path": [], "text": "retry error"},
        ]
        results = BM25Index(chunks).search(
            "retry key error", top_k=2, max_per_source=1
        )
        self.assertEqual({"D01", "D02"}, {result.source_id for result in results})


if __name__ == "__main__":
    unittest.main()
