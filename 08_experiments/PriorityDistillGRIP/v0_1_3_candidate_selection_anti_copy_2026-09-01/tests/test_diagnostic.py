import unittest

from priority_distill.records import build_evaluation_prompt
from priority_distill.supervision import build_oracle_evidence_prompt


class DiagnosticPromptTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "text": "Which entity is reached by r1 then r2?",
            "answer": "entity_c",
            "path_nodes": ["entity_a", "entity_b", "entity_c"],
            "path_relations": ["r1", "r2"],
        }

    def test_graph_free_condition_has_no_evidence(self):
        prompt = build_evaluation_prompt(self.row["text"])
        self.assertNotIn("Candidate evidence:", prompt)
        self.assertIn(self.row["text"], prompt)

    def test_oracle_evidence_condition_contains_gold_path(self):
        prompt = build_oracle_evidence_prompt(self.row)
        self.assertIn("Candidate evidence:", prompt)
        self.assertIn("entity_a --r1--> entity_b --r2--> entity_c", prompt)


if __name__ == "__main__":
    unittest.main()
