"""CPU-only regression checks for the FB15K-237 preparation contract."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_fb15k237.py"
SPEC = importlib.util.spec_from_file_location("prepare_fb15k237", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareFB15K237Test(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        entities = {
            "head": {"label": " Head. ", "description": "Description..."},
            "tail": {"label": "Tail", "description": None},
            "empty": {"label": None, "description": None},
            "description_only": {"label": None, "description": " Description. "},
            "eval_only": {"label": "Held out", "description": "Entity."},
        }
        relations = {f"r{i}": f"relation {11 - i:02d}" for i in range(12)}
        # The real dataset has 237 IDs but 233 texts: deduplicate by TEXT.
        relations["r11"] = relations["r0"]
        (raw / "entity2wikidata.json").write_text(json.dumps(entities), encoding="utf-8")
        (raw / "rel2text.json").write_text(json.dumps(relations), encoding="utf-8")
        # Unknown train entities must append in encounter order, as official.
        train = [f"head r{i % 12} tail" for i in range(24)] + ["unknown r0 head"]
        valid = [f"eval_only r{i % 12} head" for i in range(19)]
        test = [f"tail r{(i + 3) % 12} eval_only" for i in range(21)]
        for name, rows in (("train", train), ("valid", valid), ("test", test)):
            (raw / f"{name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return raw

    def prepare(self, raw, **overrides):
        kwargs = {"train_questions": 8, "eval_questions": 7, "smoke_questions": 3}
        kwargs.update(overrides)
        return MODULE.prepare_fb15k237(raw, **kwargs)

    def test_replays_official_rng_with_sorted_text_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            outputs, manifest = self.prepare(raw)
            mapping = json.loads((raw / "rel2text.json").read_text())
            # Independently replay the exact original NumPy expression order,
            # including the valid selection draw BEFORE test candidate draws.
            rng = np.random.RandomState(2026)
            pool = sorted(set(mapping.values()))
            for split, filename, out_name in (("validation", "valid.txt", "processed_val.json"), ("test", "test.txt", "processed_test.json")):
                raw_rows = [line.split() for line in (raw / filename).read_text().splitlines()]
                expected = []
                for _, relation, _ in raw_rows:
                    gold = mapping[relation]
                    remaining = pool.copy()
                    remaining.remove(gold)
                    negatives = rng.permutation(remaining)[:9]
                    expected.append(rng.permutation(negatives.tolist() + [gold]).tolist())
                indices = rng.choice(len(expected), 7, replace=False)
                samples = [q for q in outputs["recurrent_relation_prediction_full.json"]["recurrent_questions"] if q["split"] == split]
                self.assertEqual([q["source_triple_index"] for q in samples], indices.tolist())
                self.assertEqual([q["candidate_relations"] for q in samples], [expected[i] for i in indices])
                self.assertEqual(outputs[out_name]["answers"], [q["answer"] for q in samples])
            self.assertEqual(manifest["relation_order"], pool)

    def test_train_rng_cannot_perturb_eval_or_existing_train_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            first, _ = self.prepare(raw, train_questions=6)
            second, _ = self.prepare(raw, train_questions=12)
            for filename in ("processed_val.json", "processed_test.json"):
                self.assertEqual(first[filename], second[filename])
            first_q = first["recurrent_relation_prediction_full.json"]["recurrent_questions"]
            second_q = second["recurrent_relation_prediction_full.json"]["recurrent_questions"]
            self.assertEqual(first_q[:6], second_q[:6])
            self.assertEqual(first_q[6:], second_q[12:])

    def test_graph_entity_text_source_ids_and_smoke_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            outputs, manifest = self.prepare(raw)
            full = outputs["recurrent_relation_prediction_full.json"]
            smoke = outputs["recurrent_relation_prediction_smoke.json"]
            self.assertEqual(len(full["graph"]["edge_list"]), 25)
            self.assertEqual(full["graph"]["node_list"], ["Head. , Description", "Tail,", "missing", "Description", "Held out, Entity", "Unknown"])
            self.assertTrue(all(edge[0] != "Held out, Entity" and edge[2] != "Held out, Entity" for edge in full["graph"]["edge_list"]))
            self.assertEqual(len(outputs["processed_train.json"]["questions"]), 8)
            full_by_id = {q["question_id"]: q for q in full["recurrent_questions"]}
            self.assertEqual(len(smoke["recurrent_questions"]), 9)
            for q in smoke["recurrent_questions"]:
                self.assertEqual(q, full_by_id[q["question_id"]])
                self.assertEqual(len(set(q["candidate_relations"])), 10)
                self.assertEqual(q["candidate_relations"].count(q["answer"]), 1)
                self.assertEqual(q["question_id"], f"fb15k237:{q['split']}:{q['source_triple_index']}")
            self.assertEqual(manifest["counts"]["train_unknown_entities"], 1)
            repeated, repeated_manifest = self.prepare(raw)
            self.assertEqual(outputs, repeated)
            self.assertEqual(manifest, repeated_manifest)

    def test_atomic_writer_verifies_identical_and_replaces_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            payload = {"questions": ["Unicode: 中文"], "id": "0"}
            info, status = MODULE.atomic_write_json(path, payload, jsonl=True)
            self.assertEqual(status, "written")
            original_mtime = path.stat().st_mtime_ns
            self.assertEqual(len(path.read_text().splitlines()), 1)
            repeated, status = MODULE.atomic_write_json(path, payload, jsonl=True)
            self.assertEqual(info, repeated)
            self.assertEqual(status, "verified_identical")
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            path.write_text("corrupt")
            repaired, status = MODULE.atomic_write_json(path, payload, jsonl=True)
            self.assertEqual(info, repaired)
            self.assertEqual(status, "written")
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_unmapped_eval_entity_is_rejected_without_graph_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            with (raw / "valid.txt").open("a") as stream:
                stream.write("never_seen r0 head\n")
            with self.assertRaisesRegex(ValueError, "held-out entities absent"):
                self.prepare(raw)


if __name__ == "__main__":
    unittest.main()
