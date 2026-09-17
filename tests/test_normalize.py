from __future__ import annotations

import unittest

from payrag.normalize import (
    build_chunks,
    parse_html_document,
    redact_sensitive_values,
)


ARTICLE = {
    "article": {
        "content": {
            "$$mdtype": "Tag",
            "name": "Document",
            "attributes": {},
            "children": [
                {
                    "$$mdtype": "Tag",
                    "name": "Heading",
                    "attributes": {"level": 2, "id": "retry"},
                    "children": ["Retry safely"],
                },
                {
                    "$$mdtype": "Tag",
                    "name": "Callout",
                    "attributes": {"type": "caution"},
                    "children": [
                        {
                            "$$mdtype": "Tag",
                            "name": "Paragraph",
                            "attributes": {},
                            "children": ["Keep the original ", {"$$mdtype": "Tag", "name": "InlineCode", "attributes": {"content": "key"}, "children": []}, "."],
                        }
                    ],
                },
                {
                    "$$mdtype": "Tag",
                    "name": "Table",
                    "attributes": {},
                    "children": [
                        {
                            "$$mdtype": "Tag",
                            "name": "TableRow",
                            "attributes": {},
                            "children": [
                                {"$$mdtype": "Tag", "name": "TableHeaderCell", "attributes": {}, "children": ["Status"]},
                                {"$$mdtype": "Tag", "name": "TableHeaderCell", "attributes": {}, "children": ["Meaning"]},
                            ],
                        },
                        {
                            "$$mdtype": "Tag",
                            "name": "TableRow",
                            "attributes": {},
                            "children": [
                                {"$$mdtype": "Tag", "name": "TableCell", "attributes": {}, "children": ["pending"]},
                                {"$$mdtype": "Tag", "name": "TableCell", "attributes": {}, "children": ["Not complete"]},
                            ],
                        },
                    ],
                },
                {
                    "$$mdtype": "Tag",
                    "name": "ElementList",
                    "attributes": {"title": "Attributes"},
                    "children": [
                        {
                            "$$mdtype": "Tag",
                            "name": "Element",
                            "attributes": {"name": "data", "type": "object"},
                            "children": [
                                {"$$mdtype": "Tag", "name": "Paragraph", "attributes": {}, "children": ["Payload."]},
                                {
                                    "$$mdtype": "Tag",
                                    "name": "Element",
                                    "attributes": {"name": "object", "type": "object"},
                                    "children": [
                                        {"$$mdtype": "Tag", "name": "Paragraph", "attributes": {}, "children": ["Changed object."]}
                                    ],
                                },
                            ],
                        }
                    ],
                },
                {
                    "$$mdtype": "Tag",
                    "name": "CodeTabGroup",
                    "attributes": {
                        "pref": "lang",
                        "items": [
                            {
                                "id": "python",
                                "title": "Python",
                                "content": {
                                    "$$mdtype": "Tag",
                                    "name": "CodeTab",
                                    "attributes": {"id": "python"},
                                    "children": [
                                        {
                                            "$$mdtype": "Tag",
                                            "name": "CodeBlock",
                                            "attributes": {"language": "python"},
                                            "children": [
                                                "client = Client('",
                                                {"$$mdtype": "Tag", "name": "ApiKey", "attributes": {"value": "must-not-appear"}, "children": []},
                                                "')",
                                            ],
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                    "children": [],
                },
            ],
        }
    }
}


class NormalizeTests(unittest.TestCase):
    def test_preserves_heading_callout_inline_code_and_table_headers(self) -> None:
        import json

        html = f"<script>window.__INITIAL_STATE__ = {json.dumps(ARTICLE)};</script>"
        blocks = parse_html_document("D99", "0" * 64, html)

        self.assertEqual(
            ["heading", "callout", "table", "field", "field", "code"],
            [block.kind for block in blocks],
        )
        self.assertIn("`key`", blocks[1].text)
        self.assertEqual("caution", blocks[1].metadata["callout_type"])
        self.assertIn("Status | Meaning", blocks[2].text)
        self.assertEqual("data.object", blocks[4].metadata["field_path"])
        self.assertIn("data.object (object)", blocks[4].text)
        self.assertEqual("python", blocks[5].metadata["language"])
        self.assertEqual("Python", blocks[5].metadata["code_tab_title"])
        self.assertIn("<redacted-api-key>", blocks[5].text)
        self.assertNotIn("must-not-appear", blocks[5].text)

    def test_chunk_ids_are_stable(self) -> None:
        import json

        html = f"<script>window.__INITIAL_STATE__ = {json.dumps(ARTICLE)};</script>"
        blocks = parse_html_document("D99", "0" * 64, html)
        first = build_chunks("D99", "0" * 64, blocks)
        second = build_chunks("D99", "0" * 64, blocks)
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])

    def test_normalized_text_redacts_secret_shapes_and_pan(self) -> None:
        import json

        article = {
            "article": {
                "content": {
                    "$$mdtype": "Tag",
                    "name": "Document",
                    "attributes": {},
                    "children": [
                        {
                            "$$mdtype": "Tag",
                            "name": "Paragraph",
                            "attributes": {},
                            "children": [
                                "key sk_test_1234567890 and card 4242 4242 4242 4242"
                            ],
                        }
                    ],
                }
            }
        }
        html = f"<script>window.__INITIAL_STATE__ = {json.dumps(article)};</script>"
        block = parse_html_document("D99", "0" * 64, html)[0]
        self.assertEqual(
            "key <redacted-secret> and card <redacted-pan>", block.text
        )


    def test_redacts_openai_and_webhook_secrets(self) -> None:
        text = (
            "openai sk-proj-abcdefghijklmnopqrstuv "
            "webhook whsec_abcdefghijklmnopqrstuv"
        )

        redacted = redact_sensitive_values(text)

        self.assertNotIn("sk-proj-", redacted)
        self.assertNotIn("whsec_", redacted)
        self.assertGreaterEqual(redacted.count("<redacted-secret>"), 2)


if __name__ == "__main__":
    unittest.main()
