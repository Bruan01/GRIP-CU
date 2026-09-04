import unittest

import torch

from priority_distill.constrained import CandidateTrieLogitsProcessor, deduplicate_candidates


class ConstrainedTests(unittest.TestCase):
    def test_processor_masks_to_valid_next_tokens_and_eos(self):
        processor = CandidateTrieLogitsProcessor(
            candidate_token_ids=[[[4, 5], [4, 6], [7]]],
            start_length=3,
            eos_token_id=2,
            pad_token_id=0,
        )
        scores = torch.zeros((1, 8))
        out = processor(torch.tensor([[9, 9, 9, 4]]), scores)
        allowed = {index for index, value in enumerate(out[0].tolist()) if value > -1e10}
        self.assertEqual(allowed, {5, 6})

        out = processor(torch.tensor([[9, 9, 9, 7]]), scores)
        allowed = {index for index, value in enumerate(out[0].tolist()) if value > -1e10}
        self.assertEqual(allowed, {2})

        out = processor(torch.tensor([[9, 9, 9, 7, 2, 0]]), scores)
        allowed = {index for index, value in enumerate(out[0].tolist()) if value > -1e10}
        self.assertEqual(allowed, {0})

    def test_processor_handles_complete_prefix_of_longer_candidate(self):
        processor = CandidateTrieLogitsProcessor(
            candidate_token_ids=[[[3], [3, 4]]],
            start_length=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        scores = torch.zeros((1, 6))
        out = processor(torch.tensor([[9, 3]]), scores)
        allowed = {index for index, value in enumerate(out[0].tolist()) if value > -1e10}
        self.assertEqual(allowed, {2, 4})

    def test_deduplicate_candidates(self):
        class Tokenizer:
            def __call__(self, value, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in value]}

        strings, ids = deduplicate_candidates(Tokenizer(), ["ab", "ab", "c"])
        self.assertEqual(strings, ["ab", "c"])
        self.assertEqual(ids, [[97, 98], [99]])


if __name__ == "__main__":
    unittest.main()
