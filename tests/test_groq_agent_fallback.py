import unittest

import utils.gemini_relief_agent as agent


class TestGroqAgentFallback(unittest.TestCase):
    def test_generate_relief_plan_returns_error_when_client_missing(self):
        original_client = agent.client
        agent.client = None
        try:
            result = agent.generate_relief_plan(
                disaster_type="Flood",
                location="Swat",
                severity="HIGH",
                authenticity="VERIFIED",
                original_report="Heavy rainfall caused flooding.",
            )
        finally:
            agent.client = original_client

        self.assertTrue(result.startswith("Agent Error:"))


if __name__ == "__main__":
    unittest.main()
