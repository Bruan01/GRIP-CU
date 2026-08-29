import unittest

import torch
from torch import nn

from grip.recurrent.executor import FixedDepthRecurrentBlock, trace_recurrence


class MetadataBlock(nn.Module):
    attention_type = "full_attention"

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class CountingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, *args, **kwargs):
        self.calls += 1
        return (hidden_states + self.scale, "final-pass-aux")


class RecurrentExecutorTest(unittest.TestCase):
    def test_same_block_is_called_k_times_and_trace_has_k_states(self):
        inner = CountingBlock()
        recurrent = FixedDepthRecurrentBlock(inner, depth=3)
        model = nn.Sequential(recurrent)
        hidden = torch.zeros(1, 2, 4)
        with trace_recurrence(model) as traced:
            output = model(hidden)
        self.assertIs(recurrent.block, inner)
        self.assertEqual(inner.calls, 3)
        self.assertTrue(torch.equal(output[0], torch.full_like(hidden, 3.0)))
        self.assertEqual(output[1], "final-pass-aux")
        self.assertEqual(len(traced.get_trace()), 3)
        self.assertTrue(torch.equal(traced.get_trace()[1], torch.full((1, 4), 2.0)))

    def test_preserves_decoder_attention_metadata(self):
        recurrent = FixedDepthRecurrentBlock(MetadataBlock(), depth=2)
        self.assertEqual(recurrent.attention_type, "full_attention")

    def test_depth_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            FixedDepthRecurrentBlock(CountingBlock(), depth=0)


if __name__ == "__main__":
    unittest.main()
