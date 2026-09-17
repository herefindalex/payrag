from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from payrag.api import ApiSettings, create_app
from payrag.generation import GenerationResult
from payrag.retrieval import BM25Index


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.index_path = root / "bm25.json"
        BM25Index(
            [
                {
                    "source_id": "D04",
                    "chunk_id": "signature-chunk",
                    "section_path": ["Request body"],
                    "text": "Verify the signature against the raw request body.",
                }
            ]
        ).save(self.index_path, snapshot_id="test")
        self.client = TestClient(
            create_app(
                ApiSettings(
                    manifest_path=Path("config/source-manifest.json"),
                    bm25_index_path=self.index_path,
                )
            )
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_context_returns_manifest_url_and_evidence(self) -> None:
        response = self.client.post(
            "/v1/context",
            json={"question": "raw request body signature", "retriever": "bm25"},
        )
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("D04", payload["evidence"][0]["source_id"])
        self.assertEqual(
            "https://docs.stripe.com/webhooks/signature",
            payload["evidence"][0]["canonical_url"],
        )

    def test_context_redacts_secret_shaped_input(self) -> None:
        response = self.client.post(
            "/v1/context",
            json={
                "question": "signature sk_test_12345678901234567890 raw body",
                "retriever": "bm25",
            },
        )
        payload = response.json()
        self.assertTrue(payload["input_redacted"])
        self.assertNotIn("sk_test_", payload["question"])

    def test_unconfigured_dense_is_operational_error(self) -> None:
        response = self.client.post(
            "/v1/context",
            json={"question": "refund", "retriever": "dense"},
        )
        self.assertEqual(503, response.status_code)
        self.assertEqual("operational_error", response.json()["status"])

    def test_unconfigured_generation_is_operational_error(self) -> None:
        response = self.client.post("/v1/generate", json={"question": "test"})
        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "provider_not_configured",
            response.json()["operational_error"]["error_type"],
        )

    def test_generation_retrieves_server_context_and_redacts_question(self) -> None:
        app = create_app(
            ApiSettings(
                manifest_path=Path("config/source-manifest.json"),
                bm25_index_path=self.index_path,
                generation_model="test-model",
            )
        )
        completed = GenerationResult(
            status="completed",
            answer={"decision": "answerable"},
            operational_error=None,
            model="test-model",
            usage={},
            response_id="response-1",
        )
        with patch("payrag.api.OpenAIResponsesGenerator") as generator:
            generator.return_value.generate.return_value = completed
            response = TestClient(app).post(
                "/v1/generate",
                json={
                    "question": "verify sk-proj-abcdefghijklmnopqrstuv",
                    "retriever": "bm25",
                },
            )

        self.assertEqual(200, response.status_code)
        context = generator.return_value.generate.call_args.args[0]
        self.assertNotIn("sk-proj-", context["question"])
        self.assertEqual("signature-chunk", context["evidence"][0]["chunk_id"])

    def test_generation_rejects_caller_supplied_context(self) -> None:
        response = self.client.post("/v1/generate", json={"context": {}})
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
