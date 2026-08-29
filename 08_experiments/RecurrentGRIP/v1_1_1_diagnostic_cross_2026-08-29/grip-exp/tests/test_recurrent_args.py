import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "arguments" / "recurrent_args.py"
_SPEC = importlib.util.spec_from_file_location("recurrent_args_for_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
RecurrentArguments = _MODULE.RecurrentArguments


class RecurrentArgumentsTest(unittest.TestCase):
    def test_defaults_define_length_ood_pilot(self):
        args = RecurrentArguments()
        self.assertEqual(args.train_hops, [1, 2])
        self.assertEqual(args.test_hops, [3, 4])
        self.assertEqual(args.adapter_control, "all")
        self.assertEqual(args.recurrent_question_types, ["StationShortestCount"])
        self.assertEqual(args.evaluation_device, "auto")
        self.assertFalse(args.require_cuda)

    def test_rejects_overlapping_train_and_test_hops(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            RecurrentArguments(train_hops=[1, 2], test_hops=[2, 3])

    def test_rejects_invalid_recurrence_depth(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            RecurrentArguments(recurrent_depth_train=0)

    def test_rejects_unknown_evaluation_device(self):
        with self.assertRaisesRegex(ValueError, "evaluation_device"):
            RecurrentArguments(evaluation_device="tpu")

    def test_stratified_context_defaults_are_disabled(self):
        args = RecurrentArguments()
        self.assertEqual(args.context_node_samples, 0)
        self.assertEqual(args.context_edge_samples, 0)
        self.assertEqual(args.context_sampling_seed, 2026)

    def test_rejects_legacy_cap_with_stratified_sampling(self):
        with self.assertRaisesRegex(ValueError, "max_context_samples"):
            RecurrentArguments(max_context_samples=256, context_edge_samples=224)

    def test_rejects_negative_context_quota(self):
        with self.assertRaisesRegex(ValueError, "context_node_samples"):
            RecurrentArguments(context_node_samples=-1)


if __name__ == "__main__":
    unittest.main()
