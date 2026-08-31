import unittest

from structured_lora.suite import apply_gate


class SuiteGateTest(unittest.TestCase):
    def _method(self, accuracy, depths, seeds=1):
        return {
            "test_accuracy": accuracy,
            "test_accuracy_by_depth": {str(i + 1): value for i, value in enumerate(depths)},
            "seed_count": seeds,
        }

    def test_preliminary_go_requires_all_registered_effects(self):
        aggregate = {
            "monolithic": self._method(0.50, [0.70, 0.55, 0.40, 0.35]),
            "flat_oracle": self._method(0.48, [0.68, 0.52, 0.38, 0.34]),
            "ordered_prefix": self._method(0.56, [0.70, 0.58, 0.47, 0.43]),
            "permuted_depth_prefix": self._method(0.49, [0.65, 0.50, 0.42, 0.39]),
            "non_nested_random_masks": self._method(0.51, [0.68, 0.52, 0.43, 0.41]),
        }
        config = {"gate": {
            "minimum_seeds_for_final": 3,
            "ordered_over_monolithic": 0.0,
            "ordered_over_flat": 0.0,
            "deep_3_4_improvement": 0.03,
            "maximum_1hop_drop": 0.01,
            "ordered_over_permuted": 0.0,
            "ordered_over_non_nested": 0.0,
        }}
        result = apply_gate(aggregate, config)
        self.assertEqual(result["status"], "PRELIMINARY_GO")
        self.assertTrue(result["all_checks_pass"])

    def test_final_failure_stops_router(self):
        aggregate = {
            "monolithic": self._method(0.55, [0.70, 0.60, 0.48, 0.42], seeds=3),
            "flat_oracle": self._method(0.52, [0.68, 0.57, 0.45, 0.38], seeds=3),
            "ordered_prefix": self._method(0.54, [0.70, 0.59, 0.47, 0.40], seeds=3),
            "permuted_depth_prefix": self._method(0.50, [0.65, 0.55, 0.44, 0.36], seeds=3),
            "non_nested_random_masks": self._method(0.51, [0.67, 0.56, 0.45, 0.37], seeds=3),
        }
        config = {"gate": {
            "minimum_seeds_for_final": 3,
            "ordered_over_monolithic": 0.0,
            "ordered_over_flat": 0.0,
            "deep_3_4_improvement": 0.03,
            "maximum_1hop_drop": 0.01,
            "ordered_over_permuted": 0.0,
            "ordered_over_non_nested": 0.0,
        }}
        result = apply_gate(aggregate, config)
        self.assertEqual(result["status"], "STOP_STRUCTURED_LORA")
        self.assertFalse(result["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()
