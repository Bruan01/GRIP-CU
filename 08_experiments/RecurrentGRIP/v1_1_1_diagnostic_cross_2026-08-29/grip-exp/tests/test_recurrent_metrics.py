import importlib.util
import sys
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_EVALUATION_PACKAGE = types.ModuleType("evaluation")
_EVALUATION_PACKAGE.__path__ = [str(_ROOT / "evaluation")]
sys.modules.setdefault("evaluation", _EVALUATION_PACKAGE)
_UTILS_SPEC = importlib.util.spec_from_file_location("evaluation.utils", _ROOT / "evaluation" / "utils.py")
_UTILS_MODULE = importlib.util.module_from_spec(_UTILS_SPEC)
assert _UTILS_SPEC.loader is not None
_UTILS_SPEC.loader.exec_module(_UTILS_MODULE)
sys.modules["evaluation.utils"] = _UTILS_MODULE
_METRICS_SPEC = importlib.util.spec_from_file_location(
    "evaluation.recurrent_metrics", _ROOT / "evaluation" / "recurrent_metrics.py"
)
_METRICS_MODULE = importlib.util.module_from_spec(_METRICS_SPEC)
assert _METRICS_SPEC.loader is not None
_METRICS_SPEC.loader.exec_module(_METRICS_MODULE)
summarize_recurrent_predictions = _METRICS_MODULE.summarize_recurrent_predictions


class RecurrentMetricsTest(unittest.TestCase):
    def test_best_k_correlation_excludes_questions_never_solved(self):
        rows = [
            {"question_id": "q1", "true_hop": 1, "recurrence_k": 1, "adapter_control": "correct", "correct": True},
            {"question_id": "q1", "true_hop": 1, "recurrence_k": 4, "adapter_control": "correct", "correct": True},
            {"question_id": "q2", "true_hop": 4, "recurrence_k": 1, "adapter_control": "correct", "correct": False},
            {"question_id": "q2", "true_hop": 4, "recurrence_k": 4, "adapter_control": "correct", "correct": True},
            {"question_id": "q3", "true_hop": 3, "recurrence_k": 1, "adapter_control": "correct", "correct": False},
            {"question_id": "q3", "true_hop": 3, "recurrence_k": 4, "adapter_control": "correct", "correct": False},
        ]
        summary = summarize_recurrent_predictions(rows)
        self.assertEqual(summary["solved_question_count"], 2)
        self.assertEqual(summary["unsolved_question_count"], 1)
        self.assertEqual(summary["best_k_true_hop_spearman"], 1.0)

    def test_unknown_hops_remain_in_accuracy_but_not_hop_correlation(self):
        rows = [
            {"question_id": "q1", "true_hop": None, "recurrence_k": 1, "adapter_control": "correct", "correct": True, "metadata": {"split": "test"}},
            {"question_id": "q1", "true_hop": None, "recurrence_k": 2, "adapter_control": "correct", "correct": False, "metadata": {"split": "test"}},
            {"question_id": "q2", "true_hop": 3, "recurrence_k": 1, "adapter_control": "correct", "correct": False, "metadata": {"split": "validation"}},
            {"question_id": "q2", "true_hop": 3, "recurrence_k": 2, "adapter_control": "correct", "correct": True, "metadata": {"split": "validation"}},
        ]
        summary = summarize_recurrent_predictions(rows)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["known_hop_count"], 2)
        self.assertEqual(summary["unknown_hop_count"], 2)
        self.assertEqual(len(summary["by_k_and_adapter"]), 2)
        self.assertEqual(len(summary["by_split_k_and_adapter"]), 4)
        self.assertTrue(all("hop=3" in key for key in summary["by_hop_and_k"]))

    def test_cross_depth_buckets_and_k1_k2_transitions_are_separated_by_train_depth(self):
        rows = [
            self._row("q1", 1, 1, True),
            self._row("q1", 1, 2, False),
            self._row("q2", 1, 1, False),
            self._row("q2", 1, 2, True),
            self._row("q1", 2, 1, True),
            self._row("q1", 2, 2, True),
        ]
        summary = summarize_recurrent_predictions(rows)

        self.assertEqual(
            summary["by_train_k_eval_k_and_adapter"][
                "train_k=1|eval_k=1|adapter=correct"
            ]["accuracy"],
            0.5,
        )
        self.assertEqual(
            summary["by_split_train_k_eval_k_and_adapter"][
                "split=test|train_k=2|eval_k=2|adapter=correct"
            ]["accuracy"],
            1.0,
        )
        train1 = summary["k1_k2_transitions"][
            "split=test|train_k=1|adapter=correct"
        ]
        self.assertEqual(train1["paired_count"], 2)
        self.assertEqual(train1["k1_correct_k2_wrong"], 1)
        self.assertEqual(train1["k1_wrong_k2_correct"], 1)
        train2 = summary["k1_k2_transitions"][
            "split=test|train_k=2|adapter=correct"
        ]
        self.assertEqual(train2["both_correct"], 1)

    def test_output_quality_reports_empty_candidate_eos_and_token_rates(self):
        rows = [
            self._row(
                "q1", 1, 1, True, response="r1", raw_response="<answer>r1</answer>",
                generated_token_count=4, ended_with_eos=True, response_in_candidates=True,
            ),
            self._row(
                "q2", 1, 1, False, response="", raw_response="",
                generated_token_count=8, ended_with_eos=False, response_in_candidates=False,
            ),
        ]
        quality = summarize_recurrent_predictions(rows)["output_quality"]["overall"]

        self.assertEqual(quality["count"], 2)
        self.assertEqual(quality["empty_response_rate"], 0.5)
        self.assertEqual(quality["candidate_exact_rate"], 0.5)
        self.assertEqual(quality["candidate_out_of_set_rate"], 0.5)
        self.assertEqual(quality["eos_rate"], 0.5)
        self.assertEqual(quality["mean_generated_token_count"], 6.0)

    def test_state_dynamics_reports_norm_growth_and_direction_change(self):
        rows = [
            self._row(
                "q1", 2, 2, True,
                step_pooled_hidden_states=[[3.0, 4.0], [0.0, 10.0]],
            ),
            self._row(
                "q2", 2, 2, False,
                step_pooled_hidden_states=[[1.0, 0.0], [1.0, 0.0]],
            ),
        ]
        dynamics = summarize_recurrent_predictions(rows)["state_dynamics"]
        bucket = dynamics["by_train_k_eval_k_and_adapter"][
            "train_k=2|eval_k=2|adapter=correct"
        ]

        self.assertEqual(bucket["trace_present_count"], 2)
        self.assertEqual(bucket["transition_row_count"], 2)
        self.assertEqual(bucket["mean_trace_length"], 2.0)
        self.assertAlmostEqual(bucket["mean_initial_hidden_norm"], 3.0)
        self.assertAlmostEqual(bucket["mean_final_hidden_norm"], 5.5)
        self.assertAlmostEqual(bucket["mean_final_initial_norm_ratio"], 1.5)
        self.assertAlmostEqual(bucket["mean_consecutive_cosine"], 0.9)
        self.assertAlmostEqual(
            bucket["mean_consecutive_relative_delta"],
            ((45.0 ** 0.5) / 5.0) / 2.0,
        )

    @staticmethod
    def _row(question_id, train_k, eval_k, correct, **extra):
        row = {
            "graph_id": "nell23k",
            "question_id": question_id,
            "true_hop": None,
            "recurrence_k": eval_k,
            "recurrent_train_k": train_k,
            "adapter_control": "correct",
            "correct": correct,
            "response": "r1" if correct else "r2",
            "raw_response": "r1" if correct else "r2",
            "metadata": {"split": "test", "candidate_relations": ["r1", "r2"]},
        }
        row.update(extra)
        return row


if __name__ == "__main__":
    unittest.main()
