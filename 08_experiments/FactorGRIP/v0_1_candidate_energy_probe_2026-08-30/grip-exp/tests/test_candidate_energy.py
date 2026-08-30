import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from candidate_energy.artifacts import completed_question_ids, prepare_run_dir
from candidate_energy.constrained import CandidateTrieLogitsProcessor, build_candidate_trie
from transformers import LogitsProcessorList
from candidate_energy.probe import _argmax_from_dicts
from candidate_energy.scoring import CandidateScore, compute_candidate_logprobs
from scripts.run_candidate_energy_probe import parse_args


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 50 + 1 for ch in text]


class CandidateEnergyTest(unittest.TestCase):
    def test_candidate_sorting_and_length_normalisation(self):
        scores = [CandidateScore("short", -2.0, 2), CandidateScore("long", -3.0, 6)]
        self.assertEqual(_argmax_from_dicts([s.__dict__ | {"norm_logprob": s.norm_logprob} for s in scores], True)[0], 1)
        self.assertEqual(scores[0].norm_logprob, -1.0)
        self.assertEqual(scores[1].norm_logprob, -0.5)

    def test_logprob_gather_uses_next_token_positions(self):
        # Make token 2 likely at position 1 and token 3 likely at position 2.
        logits = torch.full((1, 4, 5), -10.0)
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        input_ids = torch.tensor([[8, 1, 2, 3]])
        result = compute_candidate_logprobs(logits, input_ids, [(2, 4)])[0]
        self.assertEqual(result.num_tokens, 2)
        self.assertGreater(result.sum_logprob, -0.01)

    def test_target_is_not_used_by_probe_input_construction(self):
        source = inspect.getsource(__import__("candidate_energy.probe", fromlist=["run_candidate_energy_probe"]))
        self.assertNotIn("target", source[source.index("def _build_eval_dataset"):source.index("def _input_device")])

    def test_candidate_order_permutation_keeps_argmax_candidate(self):
        rows = [{"candidate": "a", "norm_logprob": -0.2}, {"candidate": "b", "norm_logprob": -0.8}]
        self.assertEqual(rows[_argmax_from_dicts(rows, True)[0]]["candidate"], "a")
        permuted = list(reversed(rows))
        self.assertEqual(permuted[_argmax_from_dicts(permuted, True)[0]]["candidate"], "a")

    def test_trie_processor_masks_non_candidates(self):
        trie = build_candidate_trie(TinyTokenizer(), ["r1"])
        processor = CandidateTrieLogitsProcessor(trie, 2, 99, 60)
        scores = torch.zeros((1, 60))
        output = processor(torch.tensor([[7, 8]]), scores)
        self.assertTrue(torch.isfinite(output).any())
        self.assertEqual(int(torch.isfinite(output).sum()), 1)

    def test_trie_processor_is_compatible_with_transformers_processor_list(self):
        trie = build_candidate_trie(TinyTokenizer(), ["r1"])
        processor = CandidateTrieLogitsProcessor(trie, 2, 99, 60)
        scores = torch.zeros((1, 60))
        output = LogitsProcessorList([processor])(torch.tensor([[7, 8]]), scores)
        self.assertEqual(int(torch.isfinite(output).sum()), 1)

    def test_cli_accepts_hyphen_and_underscore_local_files_flags(self):
        common = [
            "run_candidate_energy_probe.py",
            "--input_file", "input.json",
            "--adapter_k1", "k1",
            "--adapter_k2", "k2",
        ]
        for spelling in ("--local-files-only", "--local_files_only"):
            with self.subTest(spelling=spelling):
                old_argv = sys.argv
                try:
                    sys.argv = common + [spelling]
                    self.assertTrue(parse_args().local_files_only)
                finally:
                    sys.argv = old_argv

    def test_resume_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = prepare_run_dir(root, "run_a")
            (run / "config.json").write_text(json.dumps({"run_id": "run_a"}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_run_dir(root, "run_a")
            prepare_run_dir(root, "run_a", resume=True)
            pred = run / "predictions.jsonl"
            pred.write_text(json.dumps({"question_id": "q", "split": "test", "adapter_control": "none", "decoder_type": "free"}) + "\n", encoding="utf-8")
            self.assertIn(("q", "test", "none", "free"), completed_question_ids(pred))


if __name__ == "__main__":
    unittest.main()
