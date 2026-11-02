"""CPU regression tests for official WN18RR preparation and safe downloads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_wn18rr as prep
import download_wn18rr as downloader


class PrepareWN18RRTest(unittest.TestCase):
    def fixture(self, root):
        raw = root / "raw"
        raw.mkdir()
        (raw / "entity2text.txt").write_text("head\tHead, first sense.\ntail\tTail.\neval\tHeld out text\n", encoding="utf-8")
        # Train includes duplicates and an unmapped training node; validation
        # uses a mapped entity never present in train edges. No trailing LF.
        self.rows = {
            "train": [("head", f"_relation_{i % 11:02d}", "tail") for i in range(22)] + [("unknown", "_relation_00", "head")],
            "validation": [("eval", f"_relation_{i % 11:02d}", "head") for i in range(13)],
            "test": [("tail", f"_relation_{(i + 3) % 11:02d}", "eval") for i in range(15)],
        }
        for split, name in prep.SPLIT_FILES.items():
            (raw / name).write_text("\n".join("\t".join(row) for row in self.rows[split]), encoding="utf-8")
        return raw

    def prepare(self, raw, **overrides):
        kwargs = dict(train_questions=8, smoke_questions=3, validate_counts=False)
        kwargs.update(overrides)
        return prep.prepare_wn18rr(raw, **kwargs)

    def test_official_all11_rng_schedule_full_evaluation_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            outputs, manifest = self.prepare(raw)
            pool = [f"relation {i:02d}" for i in range(11)]
            rng = np.random.RandomState(2026)
            full = outputs["recurrent_relation_prediction_full.json"]
            for split in ("validation", "test"):
                questions = [q for q in full["recurrent_questions"] if q["split"] == split]
                self.assertEqual(len(questions), len(self.rows[split]))
                for index, (q, (src, rel, tgt)) in enumerate(zip(questions, self.rows[split])):
                    candidates = rng.permutation(pool).tolist()
                    self.assertEqual(q["candidate_relations"], candidates)
                    self.assertEqual(set(candidates), set(pool))
                    self.assertEqual(q["question_id"], f"wn18rr:{split}:{index}")
                    self.assertEqual((q["source_entity_id"], q["source_relation_id"], q["target_entity_id"]), (src, rel, tgt))
                    self.assertEqual(q["answer"], " ".join(rel.split("_")[1:]))
                    expected = f"What is the relation between word node {q['source_node']} and word node {q['target_node']}? Selected from the following candidate answers: {'; '.join(candidates)}."
                    self.assertEqual(q["question"], expected)
            self.assertEqual(manifest["relation_order"], pool)
            self.assertFalse(manifest["eval_subsampling"])
            self.assertEqual(manifest["counts"]["full_questions"], {"train": 8, "validation": 13, "test": 15})

    def test_train_selection_is_independent_and_graph_is_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            first, manifest = self.prepare(raw, train_questions=5)
            second, _ = self.prepare(raw, train_questions=12)
            for name in ("processed_val.json", "processed_test.json"):
                self.assertEqual(first[name], second[name])
            q1 = first["recurrent_relation_prediction_full.json"]["recurrent_questions"]
            q2 = second["recurrent_relation_prediction_full.json"]["recurrent_questions"]
            self.assertEqual(q1[:5], q2[:5])
            self.assertEqual([q["source_triple_index"] for q in q1[:5]], np.random.RandomState(2026).permutation(23)[:5].tolist())
            graph = first["processed_train.json"]["graph"]
            self.assertEqual(graph["node_list"], ["Head, first sense.", "Tail.", "Held out text", "Unknown"])
            self.assertEqual(len(graph["edge_index"]), 23)
            self.assertEqual(graph["edge_index"][-1], [3, 0])
            self.assertEqual(graph["edge_index"][:22], [[0, 1]] * 22)
            self.assertTrue(all(2 not in edge for edge in graph["edge_index"]))
            self.assertEqual(manifest["counts"]["train_unknown_entities"], 1)

    def test_output_hashes_manifest_relation_helper_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            out = root / "out"
            kwargs = dict(train_questions=8, smoke_questions=3, validate_counts=False)
            manifest, statuses = prep.write_preparation(raw, out, **kwargs)
            self.assertTrue(all(status == "written" for status in statuses.values()))
            paths = list(out.iterdir()) + [raw / "rel2text.json"]
            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            repeated, statuses = prep.write_preparation(raw, out, **kwargs)
            self.assertEqual(manifest, repeated)
            self.assertTrue(all(status == "verified_identical" for status in statuses.values()))
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in paths})
            for name, digest in manifest["source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((raw / name).read_bytes()).hexdigest())
            self.assertEqual(len(manifest["outputs"]), 5)
            for name, info in manifest["outputs"].items():
                content = (out / name).read_bytes()
                self.assertEqual(info, {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
                self.assertEqual(len(content.splitlines()), 1)
                json.loads(content)
            self.assertEqual(manifest["raw_outputs"]["rel2text.json"]["sha256"], prep.sha256_file(raw / "rel2text.json"))
            mapping = json.loads((raw / "rel2text.json").read_text())
            self.assertEqual(mapping["_relation_10"], "relation 10")
            full = json.loads((out / "recurrent_relation_prediction_full.json").read_text())
            smoke = json.loads((out / "recurrent_relation_prediction_smoke.json").read_text())
            expected = [q for split in prep.SPLIT_FILES for q in [item for item in full["recurrent_questions"] if item["split"] == split][:3]]
            self.assertEqual(smoke["recurrent_questions"], expected)
            (out / "processed_test.json").write_text("corrupt")
            _, repaired = prep.write_preparation(raw, out, **kwargs)
            self.assertEqual(repaired["processed_test.json"], "written")
            self.assertEqual(list(out.glob(".*.tmp")), [])

    def test_official_sized_sources_keep_full_eval_and_default_2000_references(self):
        # Synthetic records at real split sizes exercise the production defaults
        # without network access or pretending these are official source data.
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            for split, name in prep.SPLIT_FILES.items():
                rows = self.rows[split]
                (raw / name).write_text("\n".join("\t".join(rows[i % len(rows)]) for i in range(prep.EXPECTED_COUNTS[split])))
            outputs, manifest = prep.prepare_wn18rr(raw)
            self.assertTrue(manifest["official_counts_validated"])
            self.assertEqual(manifest["counts"]["source_triples"], {"train": 86835, "validation": 3034, "test": 3134})
            self.assertEqual(manifest["counts"]["full_questions"], {"train": 2000, "validation": 3034, "test": 3134})
            self.assertEqual(len(outputs["processed_val.json"]["questions"]), 3034)
            self.assertEqual(len(outputs["processed_test.json"]["questions"]), 3134)
            report = downloader.verify(raw)
            self.assertEqual(report["relations_in_splits"], 11)
            self.assertEqual(report["source_sha256"], manifest["source_sha256"])

    def test_cli_defaults_reject_incomplete_source_and_fixture_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = self.fixture(root)
            command = [sys.executable, str(SCRIPTS / "prepare_wn18rr.py"), "--raw_dir", str(raw), "--out_dir", str(root / "out"), "--train_questions", "8", "--smoke_questions", "3"]
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("expected 86835 triples", failed.stderr)
            self.assertFalse((root / "out").exists())
            success = subprocess.run(command + ["--allow_small_fixture"], capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["counts"]["full_questions"]["test"], 15)

    def test_rejects_pointer_malformed_blank_and_duplicate_entity_sources(self):
        cases = [
            ("train.txt", "version https://git-lfs.github.com/spec/v1\noid sha256:bad\nsize 1\n", "Git LFS pointer"),
            ("entity2text.txt", "version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
            ("test.txt", "head _relation_00 tail extra", "exactly three"),
            ("valid.txt", "head _relation_00 tail\n\n", "blank source"),
            ("entity2text.txt", "head\tOne\nhead\tTwo", "duplicate entity"),
        ]
        for name, content, message in cases:
            with self.subTest(name=name, message=message), tempfile.TemporaryDirectory() as directory:
                raw = self.fixture(Path(directory))
                (raw / name).write_text(content)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(raw)

    def test_heldout_entity_relation_coverage_and_exact_relation_count(self):
        for row, message in [("missing _relation_00 head", "held-out entities absent"), ("head _held_out tail", "relations absent")]:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                raw = self.fixture(Path(directory))
                (raw / "valid.txt").write_text(row)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(raw)
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            path = raw / "train.txt"
            path.write_text(path.read_text().replace("_relation_10", "_relation_09"))
            with self.assertRaisesRegex(ValueError, "11 distinct"):
                self.prepare(raw)

    def test_verifier_detects_provenance_candidates_and_eval_dropping(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            outputs, _ = self.prepare(raw)
            entities, triples, relations, _ = prep.load_raw(raw, validate_counts=False)
            entity_ids = {key: index for index, key in enumerate(entities)}
            for mutation in ("source", "candidates", "drop"):
                with self.subTest(mutation=mutation):
                    altered = copy.deepcopy(outputs)
                    questions = altered["recurrent_relation_prediction_full.json"]["recurrent_questions"]
                    if mutation == "source":
                        questions[-1]["source_entity_id"] = "wrong"
                    elif mutation == "candidates":
                        questions[-1]["candidate_relations"][0] = "bogus"
                    else:
                        questions.pop()
                    with self.assertRaises(ValueError):
                        prep.verify_records(altered, triples, entities, entity_ids, relations, 8)

    def test_changing_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = self.fixture(Path(directory))
            real_hash = prep.sha256_file
            counts = {}
            def changed_hash(path):
                counts[path.name] = counts.get(path.name, 0) + 1
                return "changed" if path.name == "test.txt" and counts[path.name] > 1 else real_hash(path)
            with patch.object(prep, "sha256_file", side_effect=changed_hash):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    self.prepare(raw)

    def test_downloader_stages_validates_repairs_and_preserves_existing_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "entity2text.txt"
            valid = b"head\tWord text\n"
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(valid)) as request:
                downloader.download("https://example.invalid/source", dest, timeout=7)
                self.assertEqual(request.call_args.kwargs["timeout"], 7)
            self.assertEqual(dest.read_bytes(), valid)
            with patch.object(downloader.urllib.request, "urlopen") as request:
                downloader.download("https://example.invalid/source", dest)
                request.assert_not_called()
            dest.write_text("version https://git-lfs.github.com/spec/v1\n")
            old = dest.read_bytes()
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(old)):
                with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                    downloader.download("https://example.invalid/source", dest)
            self.assertEqual(dest.read_bytes(), old)
            self.assertEqual(list(dest.parent.glob("*.partial")), [])
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(valid)):
                downloader.download("https://example.invalid/source", dest)
            self.assertEqual(dest.read_bytes(), valid)

    def test_downloader_refreshes_expired_mirror_redirect_once(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "entity2text.txt"
            expired = urllib.error.HTTPError("https://example.invalid", 403, "expired", None, None)
            with patch.object(downloader.urllib.request, "urlopen", side_effect=[expired, io.BytesIO(b"head\tWord\n")]) as request:
                downloader.download("https://example.invalid/source", dest, timeout=5)
            self.assertEqual(request.call_count, 2)
            self.assertIn("cache_bust=", request.call_args.args[0].full_url)
            self.assertEqual(request.call_args.kwargs["timeout"], 5)
            self.assertEqual(dest.read_bytes(), b"head\tWord\n")

    def test_downloader_checks_triple_counts_and_does_not_log_secret_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "train.txt"
            with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(b"head _relation_00 tail\n")):
                with self.assertRaisesRegex(ValueError, "expected 86835"):
                    downloader.download("https://example.invalid/source", dest)
            self.assertFalse(dest.exists())
            secret = "NEVER_PRINT_THIS_SECRET"
            with patch.object(downloader.urllib.request, "urlopen", side_effect=urllib.error.URLError(secret)), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                with self.assertRaises(RuntimeError) as caught:
                    downloader.download(f"https://example.invalid/?token={secret}", dest)
            self.assertNotIn(secret, str(caught.exception) + stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
