import unittest

from utils.pipeline import analyze_report


class TestRuntimeFallbacks(unittest.TestCase):
    def test_analyze_report_returns_dict_without_crashing(self):
        result = analyze_report("Heavy flooding in Swat district after rains.")

        self.assertIsInstance(result, dict)
        self.assertIn("disaster", result)
        self.assertIn("location", result)
        self.assertIn("severity", result)
        self.assertIn("authenticity", result)
        self.assertIn("alerts", result)
        self.assertIn("briefing", result)


if __name__ == "__main__":
    unittest.main()
