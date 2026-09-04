import json
import tempfile
import unittest
from pathlib import Path

from entity_decoder.vocabulary import build_train_entity_vocabulary, read_entities, audit_answer_coverage


class VocabularyTest(unittest.TestCase):
    def test_builds_sorted_unique_vocabulary_from_train_graph_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.txt"
            train.write_text("b\tr1\tc\na\tr2\tb\n", encoding="utf-8")
            output = root / "entities.jsonl"
            audit = build_train_entity_vocabulary(train, output)
            self.assertEqual(read_entities(output), ["a", "b", "c"])
            self.assertEqual(audit["entity_count"], 3)
            self.assertEqual(audit["triple_count"], 2)
            self.assertEqual(audit["source_split"], "train")
            first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first, {"entity": "a", "index": 0})

    def test_reports_answer_coverage_without_adding_missing_answers(self):
        rows = [{"answer": "a"}, {"answer": "missing"}]
        audit = audit_answer_coverage(["a", "b"], rows)
        self.assertEqual(audit["covered"], 1)
        self.assertEqual(audit["total"], 2)
        self.assertEqual(audit["missing_answers"], ["missing"])

    def test_rejects_malformed_train_triple(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.txt"
            source.write_text("head\trelation-only\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_train_entity_vocabulary(source, Path(tmp) / "entities.jsonl")


if __name__ == "__main__":
    unittest.main()
