import unittest

from arguments.recurrent_args import RecurrentArguments


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


if __name__ == "__main__":
    unittest.main()
