from __future__ import annotations

import unittest

from payrag.answer import AnswerContractError, validate_answer_contract


def valid_answer() -> dict:
    return {
        "decision": "answerable",
        "applicable_conditions": [],
        "conclusions": [
            {
                "statement": "Reuse the original key.",
                "evidence_chunk_ids": ["chunk-1"],
                "engineering_inference": False,
            }
        ],
        "missing_information": [],
        "verification_steps": [],
        "operational_error": None,
    }


class AnswerContractTests(unittest.TestCase):
    def test_accepts_citation_from_retrieved_context(self) -> None:
        validate_answer_contract(valid_answer(), {"chunk-1"})

    def test_rejects_unretrieved_citation(self) -> None:
        answer = valid_answer()
        answer["conclusions"][0]["evidence_chunk_ids"] = ["invented"]
        with self.assertRaisesRegex(AnswerContractError, "unavailable chunks"):
            validate_answer_contract(answer, {"chunk-1"})

    def test_uncited_engineering_inference_must_be_marked(self) -> None:
        answer = valid_answer()
        answer["conclusions"][0]["evidence_chunk_ids"] = []
        with self.assertRaisesRegex(AnswerContractError, "engineering inference"):
            validate_answer_contract(answer, {"chunk-1"})


if __name__ == "__main__":
    unittest.main()
