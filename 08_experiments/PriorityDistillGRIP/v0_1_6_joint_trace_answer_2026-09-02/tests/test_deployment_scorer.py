import unittest

from scripts.score_deployment_candidates import summarize


class DeploymentScorerTests(unittest.TestCase):
    def test_summary_reports_rank_metrics(self):
        report = summarize([
            {"rank": 1, "depth_label": 1},
            {"rank": 2, "depth_label": 1},
            {"rank": 4, "depth_label": 2},
        ])
        self.assertEqual(report["count"], 3)
        self.assertAlmostEqual(report["rank1_accuracy"], 1 / 3)
        self.assertEqual(report["rank_counts"], {"1": 1, "2": 1, "3": 0, "4": 1})


if __name__ == "__main__":
    unittest.main()
