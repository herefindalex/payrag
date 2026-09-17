from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from payrag.generation import OpenAIResponsesGenerator, RAGAnswer


CONTEXT = {
    "question": "Should the same key be reused?",
    "evidence": [{"chunk_id": "chunk-1", "text": "Reuse the same key."}],
}


class _Responses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="response-1",
            output_parsed=self.parsed,
            output=[],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
        )


class GenerationTests(unittest.TestCase):
    def test_missing_provider_key_is_operational_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = OpenAIResponsesGenerator("test-model").generate(CONTEXT)
        self.assertEqual("operational_error", result.status)
        self.assertEqual(
            "provider_not_configured", result.operational_error["error_type"]
        )

    def test_structured_answer_with_valid_citation_is_accepted(self) -> None:
        parsed = RAGAnswer(
            decision="answerable",
            applicable_conditions=[],
            conclusions=[
                {
                    "statement": "Reuse the same key.",
                    "evidence_chunk_ids": ["chunk-1"],
                    "engineering_inference": False,
                }
            ],
            missing_information=[],
            verification_steps=[],
            operational_error=None,
        )
        client = SimpleNamespace(responses=_Responses(parsed))
        result = OpenAIResponsesGenerator("test-model", client=client).generate(CONTEXT)
        self.assertEqual("completed", result.status)
        self.assertEqual(30, result.usage["total_tokens"])

    def test_provider_payload_is_redacted(self) -> None:
        parsed = RAGAnswer(
            decision="answerable",
            applicable_conditions=[],
            conclusions=[
                {
                    "statement": "Reuse the same key.",
                    "evidence_chunk_ids": ["chunk-1"],
                    "engineering_inference": False,
                }
            ],
            missing_information=[],
            verification_steps=[],
            operational_error=None,
        )
        responses = _Responses(parsed)
        client = SimpleNamespace(responses=responses)
        sensitive_context = {
            "question": "Use sk-proj-abcdefghijklmnopqrstuv?",
            "evidence": [
                {
                    "chunk_id": "chunk-1",
                    "text": "Webhook whsec_abcdefghijklmnopqrstuv",
                }
            ],
        }

        result = OpenAIResponsesGenerator("test-model", client=client).generate(
            sensitive_context
        )

        self.assertEqual("completed", result.status)
        user_message = next(
            item for item in responses.kwargs["input"] if item["role"] == "user"
        )
        payload = json.loads(user_message["content"])
        self.assertNotIn("sk-proj-", payload["question"])
        self.assertNotIn("whsec_", payload["evidence"][0]["text"])

    def test_unretrieved_citation_becomes_contract_error(self) -> None:
        parsed = RAGAnswer(
            decision="answerable",
            applicable_conditions=[],
            conclusions=[
                {
                    "statement": "Unsupported.",
                    "evidence_chunk_ids": ["invented"],
                    "engineering_inference": False,
                }
            ],
            missing_information=[],
            verification_steps=[],
            operational_error=None,
        )
        client = SimpleNamespace(responses=_Responses(parsed))
        result = OpenAIResponsesGenerator("test-model", client=client).generate(CONTEXT)
        self.assertEqual("operational_error", result.status)
        self.assertEqual(
            "answer_contract_violation", result.operational_error["error_type"]
        )


if __name__ == "__main__":
    unittest.main()
