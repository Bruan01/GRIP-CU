import unittest

from entity_decoder.gate import evaluate_gate


class GateTest(unittest.TestCase):
    def _summary(self, d0, d1, novel0, novel1, raw0=None, raw1=None, valid0=.8, valid1=1.0):
        def metrics(canonical, raw, novel, valid):
            return {
                "canonical_entity_exact_match": canonical,
                "raw_exact_match": canonical if raw is None else raw,
                "valid_entity_rate": valid,
                "by_composition": {
                    "novel-composition": {"canonical_entity_exact_match": novel}
                },
            }
        return {
            "results": {
                "D0": {
                    "validation": {"metrics": metrics(d0, raw0, novel0, valid0)},
                    "test": {"metrics": metrics(.99, .99, .99, 1.0)},
                },
                "D1": {
                    "validation": {"metrics": metrics(d1, raw1, novel1, valid1)},
                    "test": {"metrics": metrics(0.0, 0.0, 0.0, 1.0)},
                },
            }
        }

    def _config(self):
        return {
            "gate": {
                "split": "validation",
                "decoder": "D1",
                "primary_checkpoints": ["direct43", "direct44"],
                "reference_checkpoints": ["more42"],
                "minimum_primary_checkpoints": 2,
                "mean_canonical_gain_pp": 2.0,
                "max_mean_novel_composition_drop_pp": 1.0,
                "require_nonnegative_per_checkpoint_raw_and_canonical": True,
            }
        }

    def test_go_when_two_direct_checkpoints_clear_validation_gate(self):
        summaries = {
            "direct43": self._summary(.30, .33, .20, .20, raw0=.29, raw1=.32),
            "direct44": self._summary(.31, .34, .24, .23, raw0=.30, raw1=.33),
            "more42": self._summary(.32, .32, .22, .22),
        }
        result = evaluate_gate(self._config(), summaries)
        self.assertEqual(result["decision"], "PRELIMINARY_GO_D2")
        self.assertAlmostEqual(result["aggregate"]["mean_canonical_gain_pp"], 3.0)
        self.assertAlmostEqual(result["aggregate"]["mean_novel_composition_gain_pp"], -0.5)
        self.assertTrue(all(result["checks"].values()))

    def test_stop_when_validity_rises_without_semantic_em_gain(self):
        summaries = {
            "direct43": self._summary(.30, .30, .20, .20, valid0=.80, valid1=1.0),
            "direct44": self._summary(.31, .31, .24, .24, valid0=.75, valid1=1.0),
            "more42": self._summary(.32, .32, .22, .22),
        }
        result = evaluate_gate(self._config(), summaries)
        self.assertEqual(result["decision"], "STOP_DECODER_PRIMARY")
        self.assertFalse(result["checks"]["mean_canonical_gain"])
        self.assertGreater(result["aggregate"]["mean_valid_rate_gain_pp"], 0)

    def test_stop_when_novel_composition_mean_drops_more_than_one_point(self):
        summaries = {
            "direct43": self._summary(.30, .33, .20, .18),
            "direct44": self._summary(.31, .34, .24, .22),
            "more42": self._summary(.32, .32, .22, .22),
        }
        result = evaluate_gate(self._config(), summaries)
        self.assertEqual(result["decision"], "STOP_DECODER_PRIMARY")
        self.assertFalse(result["checks"]["novel_composition_floor"])

    def test_stop_when_one_direct_checkpoint_raw_direction_disagrees(self):
        summaries = {
            "direct43": self._summary(.30, .33, .20, .20, raw0=.30, raw1=.29),
            "direct44": self._summary(.31, .34, .24, .24, raw0=.30, raw1=.33),
            "more42": self._summary(.32, .32, .22, .22),
        }
        result = evaluate_gate(self._config(), summaries)
        self.assertEqual(result["decision"], "STOP_DECODER_PRIMARY")
        self.assertFalse(result["checks"]["raw_and_canonical_same_direction"])

    def test_gate_uses_validation_and_ignores_test_metrics(self):
        summaries = {
            "direct43": self._summary(.30, .33, .20, .20),
            "direct44": self._summary(.31, .34, .24, .24),
            "more42": self._summary(.32, .32, .22, .22),
        }
        result = evaluate_gate(self._config(), summaries)
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["decision"], "PRELIMINARY_GO_D2")

    def test_requires_two_direct_summaries_and_reference_evidence(self):
        summaries = {
            "direct43": self._summary(.30, .33, .20, .20),
            "more42": self._summary(.32, .32, .22, .22),
        }
        with self.assertRaises(ValueError):
            evaluate_gate(self._config(), summaries)


if __name__ == "__main__":
    unittest.main()
