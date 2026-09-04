import unittest

from priority_distill.runtime_data import balance_rows_to_token_budget, tokenize_supervision_rows


class BudgetTests(unittest.TestCase):
    def test_cycles_short_rows_until_reference_budget_is_reached(self):
        rows = [{"task_id": "a", "input_token_count": 3}, {"task_id": "b", "input_token_count": 4}]
        balanced, audit = balance_rows_to_token_budget(rows, target_input_tokens=15, seed=1)
        self.assertGreaterEqual(audit["input_tokens"], 15)
        self.assertLess(audit["input_tokens"], 19)
        self.assertGreater(len(balanced), len(rows))

    def test_tokenizer_reports_prompt_truncation(self):
        class FakeTokenizer:
            eos_token = "<eos>"
            def __call__(self, text, add_special_tokens=True):
                size = 12 if text == "long prompt" else 2
                return {"input_ids": list(range(size))}
        rows = [{"task_id": "x", "prompt": "long prompt", "answer": "a", "depth_label": 1}]
        tokenized = tokenize_supervision_rows(rows, FakeTokenizer(), max_length=10)
        self.assertEqual(tokenized[0]["truncated_prompt_tokens"], 4)
        self.assertEqual(tokenized[0]["input_token_count"], 10)

    def test_budget_order_is_deterministic(self):
        rows = [{"task_id": str(index), "input_token_count": index + 1} for index in range(4)]
        left, _ = balance_rows_to_token_budget(rows, target_input_tokens=20, seed=9)
        right, _ = balance_rows_to_token_budget(rows, target_input_tokens=20, seed=9)
        self.assertEqual([row["task_id"] for row in left], [row["task_id"] for row in right])


if __name__ == "__main__":
    unittest.main()
