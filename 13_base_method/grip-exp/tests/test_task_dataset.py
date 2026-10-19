import unittest

from grip.tasks.train_tasks.task_dataset import TaskDataset


class DummyTokenizer:
    def __call__(self, text, truncation=True, padding=False):
        return {"input_ids": [1, 2]}


class TaskDatasetTest(unittest.TestCase):
    def test_candidate_metadata_is_attached_only_during_qa_stage(self):
        context = ["ctx-1", "ctx-2"]
        qa = ["qa-1"]
        candidate = {
            "title": "graph",
            "question": "relation?",
            "candidates": ["r1", "r2"],
            "positive_index": 0,
        }
        dataset = TaskDataset(context, qa, DummyTokenizer(), candidate_samples=[candidate])

        self.assertNotIn("candidate_record", dataset[0])
        dataset.enable_qa = True
        self.assertEqual(dataset.data_list, qa)
        self.assertEqual(dataset[0]["candidate_record"], candidate)

    def test_candidate_records_are_aligned_without_modulo_reuse(self):
        context = ["ctx"]
        qa = ["qa-1", "qa-2"]
        candidates = [{"id": 0}, None]
        dataset = TaskDataset(context, qa, DummyTokenizer(), candidate_samples=candidates)
        dataset.enable_qa = True

        self.assertEqual(dataset[0]["candidate_record"], candidates[0])
        self.assertIsNone(dataset[1]["candidate_record"])


if __name__ == "__main__":
    unittest.main()
