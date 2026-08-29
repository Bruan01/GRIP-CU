import unittest

from grip.recurrent.outputs import RecurrentPrediction
from scripts.run_recurrent_grip import _parse_answer


class RecurrentOutputSchemaTest(unittest.TestCase):
    def test_bracket_answer_is_parsed_without_tags(self):
        self.assertEqual(_parse_answer("[concept:teamplaysinleague]"), "concept:teamplaysinleague")

    def test_tagged_answer_is_parsed(self):
        self.assertEqual(_parse_answer("<answer>r1</answer>"), "r1")

    def test_serialized_prediction_contains_mechanism_and_control_fields(self):
        prediction = RecurrentPrediction(
            graph_id="g0",
            question_id="g0:2",
            question="How many stations are between A and D?",
            target=["2"],
            true_hop=3,
            recurrence_k=4,
            adapter_id="g1",
            adapter_control="shuffled",
            raw_response="<answer>2</answer>",
            response="2",
            correct=True,
            latency_seconds=0.1,
            step_pooled_hidden_states=[[0.0], [1.0], [2.0], [3.0]],
            metadata={"split": "test"},
        )
        prediction.validate()
        row = prediction.to_dict()
        self.assertEqual(row["true_hop"], 3)
        self.assertEqual(row["recurrence_k"], 4)
        self.assertEqual(row["adapter_id"], "g1")
        self.assertEqual(len(row["step_pooled_hidden_states"]), 4)

    def test_unknown_structural_hop_is_valid_for_nell23k(self):
        prediction = RecurrentPrediction(
            graph_id="nell23k",
            question_id="nell23k:test:1",
            question="relation?",
            target=["r1"],
            true_hop=None,
            recurrence_k=2,
            adapter_id="nell23k",
            adapter_control="correct",
            raw_response="<answer>r1</answer>",
            response="r1",
            correct=True,
            latency_seconds=0.1,
        )
        prediction.validate()
        self.assertIsNone(prediction.to_dict()["true_hop"])


if __name__ == "__main__":
    unittest.main()
