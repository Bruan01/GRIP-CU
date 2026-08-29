import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_recurrent_nell23k.py"
SPEC = importlib.util.spec_from_file_location("prepare_recurrent_nell23k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class PrepareRecurrentNELL23KTest(unittest.TestCase):
    def _raw_dir(self, root: Path) -> Path:
        raw = root / "nell23k"
        raw.mkdir()
        (raw / "entity2text.json").write_text(
            json.dumps({name: {} for name in "abcdefg"}), encoding="utf-8"
        )
        (raw / "train.txt").write_text(
            "a r1 b\n"
            "b r2 c\n"
            "c r3 d\n"
            "d r1 e\n"
            "e r2 f\n"
            "f r3 g\n",
            encoding="utf-8",
        )
        (raw / "valid.txt").write_text("a r3 c\na r2 g\n", encoding="utf-8")
        (raw / "test.txt").write_text("a r1 d\na r2 missing\n", encoding="utf-8")
        return raw

    def test_preserves_splits_candidates_and_structural_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self._raw_dir(Path(directory))
            rows, stats = MODULE.prepare_nell23k(
                raw_dir=raw,
                max_train_questions=3,
                max_validation_questions=2,
                max_test_questions=2,
                num_candidates=3,
                seed=2026,
            )
        self.assertEqual(len(rows), 1)
        record = rows[0]
        samples = record["recurrent_questions"]
        self.assertEqual([sample["split"] for sample in samples].count("train"), 3)
        self.assertEqual([sample["split"] for sample in samples].count("validation"), 2)
        self.assertEqual([sample["split"] for sample in samples].count("test"), 2)
        for sample in samples:
            self.assertIn(sample["answer"], sample["candidate_relations"])
            self.assertEqual(len(sample["candidate_relations"]), 3)
        validation = {sample["question_id"]: sample for sample in samples if sample["split"] == "validation"}
        self.assertEqual(validation["nell23k:validation:0"]["true_hop"], 2)
        unreachable = next(
            sample for sample in samples if sample["target_node"] == "missing"
        )
        self.assertIsNone(unreachable["true_hop"])
        self.assertFalse(unreachable["structural_reachable"])
        self.assertEqual(stats["graph"]["edges"], 6)

    def test_same_seed_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self._raw_dir(Path(directory))
            kwargs = dict(
                raw_dir=raw,
                max_train_questions=3,
                max_validation_questions=1,
                max_test_questions=1,
                num_candidates=3,
                seed=7,
            )
            first, first_stats = MODULE.prepare_nell23k(**kwargs)
            second, second_stats = MODULE.prepare_nell23k(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)


if __name__ == "__main__":
    unittest.main()
