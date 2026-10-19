import unittest
from types import SimpleNamespace

import torch

from grip.training.candidate_scorer import score_candidate_prompt, score_candidate_records


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None):
        token_ids = {
            "<answer>r</answer>": [4, 5],
            "<answer>long relation</answer>": [6, 7, 8, 9],
            "<answer>x</answer>": [10, 11],
        }.get(text, [1, 2, 3, 12, 13])
        if max_length is not None:
            token_ids = token_ids[:max_length]
        result = {"input_ids": token_ids}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([token_ids], dtype=torch.long)
        return result


class PositionAwareModel(torch.nn.Module):
    """Deterministic model whose answer-position log-probs are easy to inspect."""

    def __init__(self, vocabulary_size=32):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocabulary_size),
            -3.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for position in range(sequence_length):
            preferred_token = (position + 4) % self.vocabulary_size
            logits[:, position, preferred_token] = 3.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def legacy_candidate_scores(model, tokenizer, prompt_ids, candidates, max_length=None):
    """Reference implementation matching the pre-refactor training calculation."""
    target_rows = [
        torch.tensor(
            tokenizer(f"<answer>{candidate}</answer>", add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        for candidate in candidates
    ]
    max_target_length = max(target.numel() for target in target_rows)
    prompt_length = len(prompt_ids)
    if max_length is not None:
        prompt_length = min(prompt_length, max(1, max_length - max_target_length))
    prompt = torch.tensor(prompt_ids[:prompt_length], dtype=torch.long)
    input_ids = torch.full(
        (len(target_rows), prompt_length + max_target_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, target in enumerate(target_rows):
        input_ids[row, :prompt_length] = prompt
        input_ids[row, prompt_length:prompt_length + target.numel()] = target
        attention_mask[row, :prompt_length + target.numel()] = 1
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=max_target_length + 1,
    ).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.stack([
        log_probs[row, :target.numel()].gather(1, target[:, None]).squeeze(1).mean()
        for row, target in enumerate(target_rows)
    ])


class CandidateScorerTest(unittest.TestCase):
    def setUp(self):
        self.model = PositionAwareModel()
        self.tokenizer = ToyTokenizer()
        self.prompt_ids = torch.tensor([[20, 21, 22, 23]], dtype=torch.long)

    def test_old_and_new_scores_match_for_mixed_length_candidates(self):
        candidates = ["r", "long relation", "x"]
        expected = legacy_candidate_scores(
            self.model, self.tokenizer, self.prompt_ids[0].tolist(), candidates, max_length=7,
        )
        result = score_candidate_records(
            self.model,
            self.tokenizer,
            [{"_candidate_prompt_ids": self.prompt_ids[0].tolist(), "candidates": candidates}],
            max_length=7,
            score_clip=None,
            candidate_batch_size=2,
        )
        torch.testing.assert_close(result.scores[0], expected, rtol=0.0, atol=1e-6)
        self.assertEqual(result.token_lengths[0].tolist(), [2, 4, 2])

    def test_prompt_tokens_are_not_scored_and_mean_not_sum(self):
        candidate = "r"
        result = score_candidate_prompt(
            self.model,
            self.tokenizer,
            self.prompt_ids,
            [candidate],
            score_clip=None,
        )
        target = torch.tensor([4, 5], dtype=torch.long)
        input_ids = torch.tensor([[20, 21, 22, 23, 4, 5]], dtype=torch.long)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=3,
        ).logits
        answer_log_probs = torch.log_softmax(logits, dim=-1)[0, :2].gather(1, target[:, None]).squeeze(1)
        expected_mean = answer_log_probs.mean()
        self.assertAlmostEqual(float(result.scores[0][0]), float(expected_mean), places=6)
        self.assertNotAlmostEqual(float(result.scores[0][0]), float(answer_log_probs.sum()), places=4)

    def test_padding_does_not_change_short_candidate_score(self):
        short_only = score_candidate_prompt(
            self.model,
            self.tokenizer,
            self.prompt_ids,
            ["r"],
            score_clip=None,
        )
        batched = score_candidate_prompt(
            self.model,
            self.tokenizer,
            self.prompt_ids,
            ["r", "long relation"],
            score_clip=None,
        )
        torch.testing.assert_close(batched.scores[0][0], short_only.scores[0][0], rtol=0.0, atol=1e-6)
        self.assertEqual(batched.token_lengths[0].tolist(), [2, 4])

    def test_record_batch_and_single_record_have_same_scores(self):
        records = [
            {"_candidate_prompt_ids": [20, 21, 22], "candidates": ["r", "x"]},
            {"_candidate_prompt_ids": [30, 31, 32], "candidates": ["long relation", "r"]},
        ]
        batched = score_candidate_records(
            self.model, self.tokenizer, records, max_length=None, score_clip=None,
        )
        single = [
            score_candidate_records(
                self.model, self.tokenizer, [record], max_length=None, score_clip=None,
            ).scores[0]
            for record in records
        ]
        for actual, expected in zip(batched.scores, single):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
