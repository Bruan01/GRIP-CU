import unittest

import torch
from torch import nn

from scripts.run_recurrent_grip import _place_model_for_evaluation, _resolve_evaluation_device


class RecurrentEvaluationDeviceTest(unittest.TestCase):
    def test_cpu_device_is_explicitly_resolved(self):
        self.assertEqual(_resolve_evaluation_device("cpu"), torch.device("cpu"))

    def test_cuda_request_fails_when_cuda_is_missing(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available in this runtime")
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            _resolve_evaluation_device("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_evaluation_model_is_moved_to_cuda(self):
        model = nn.Linear(4, 4)
        device = _place_model_for_evaluation(model, "cuda", rank=0)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(next(model.parameters()).device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
