from __future__ import annotations

import unittest

from payrag.expansion import with_query_control


class ExpansionTests(unittest.TestCase):
    def test_empty_query_control_is_preserved(self) -> None:
        self.assertEqual(
            "https://docs.stripe.com/api/events/object?query=",
            with_query_control("https://docs.stripe.com/api/events/object", ""),
        )

    def test_named_query_control_is_encoded(self) -> None:
        self.assertEqual(
            "https://docs.stripe.com/api/payment_intents/object?query=next_action",
            with_query_control(
                "https://docs.stripe.com/api/payment_intents/object", "next_action"
            ),
        )


if __name__ == "__main__":
    unittest.main()
