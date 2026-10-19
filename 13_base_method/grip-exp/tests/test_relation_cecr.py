import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.infer_eval_nell23k import combine_cecr_scores, score_candidates

from scripts.build_cecr_hard_negative_qa import hard_negative_pool
from grip.relation_cecr import (
    build_relation_memory_from_triples,
    infer_entity_type,
    parse_question_entities,
    parse_relation_candidates,
)


class RelationCECRTest(unittest.TestCase):
    def test_parse_candidates_preserves_randomized_order_and_strips_period(self):
        question = (
            "What is the relation between word node concept_city_a and word node concept_stateorprovince_b? "
            "Selected from the following candidate answers: concept:proxyfor; concept:subpartof; "
            "concept:citylocatedinstate."
        )
        self.assertEqual(parse_relation_candidates(question), [
            "concept:proxyfor",
            "concept:subpartof",
            "concept:citylocatedinstate",
        ])
        self.assertEqual(parse_question_entities(question), ("concept_city_a", "concept_stateorprovince_b"))

    def test_parse_direct_question_has_no_candidates(self):
        self.assertEqual(
            parse_relation_candidates("What relation holds between word node concept_a and word node concept_b?"), []
        )

    def test_relation_memory_uses_only_supplied_training_triples(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.txt"
            train.write_text(
                "concept_person_alice\tconcept:worksfor\tconcept_organization_acme\n"
                "concept_city_a\tconcept:citylocatedinstate\tconcept_stateorprovince_x\n"
                "concept_city_b\tconcept:citylocatedinstate\tconcept_stateorprovince_x\n",
                encoding="utf-8",
            )
            memory = build_relation_memory_from_triples(train)
        card = memory.card("concept:citylocatedinstate")
        self.assertIsNotNone(card)
        self.assertEqual(card.frequency, 2)
        self.assertEqual(card.source_types, ("geopolitical_location",))
        self.assertEqual(card.target_types, ("geopolitical_location",))
        self.assertIn("concept:subpartof", card.confusing_relations)
        self.assertEqual(
            memory.relation_prior("concept:citylocatedinstate", "concept_city_a", "concept_stateorprovince_x"),
            1.0,
        )
        card_text = memory.render_card(
            "concept:citylocatedinstate",
            ["concept:citylocatedinstate", "concept:subpartof"],
        )
        self.assertIn("concept:citylocatedinstate", card_text)
        self.assertIn("confusing candidate relations", card_text)
        self.assertTrue(memory.evidence_detail(
            "concept:citylocatedinstate", "concept_city_a", "concept_stateorprovince_x"
        )["exact_train_edge"])
        self.assertEqual(
            memory.evidence_score(
                "concept:citylocatedinstate", "concept_city_a", "concept_stateorprovince_x"
            ),
            1.0,
        )

    def test_hard_negative_pool_prioritizes_known_confusions(self):
        memory = build_relation_memory_from_triples([
            ("concept_city_a", "concept:subpartof", "concept_stateorprovince_x"),
            ("concept_city_a", "concept:proxyfor", "concept_stateorprovince_x"),
            ("concept_city_a", "concept:citylocatedinstate", "concept_stateorprovince_x"),
        ])
        pool = hard_negative_pool(
            memory, "concept:subpartof", "concept_city_a", "concept_stateorprovince_x",
            ["concept:subpartof", "concept:proxyfor", "concept:citylocatedinstate"],
        )
        self.assertIn("concept:proxyfor", pool[:2])
        self.assertIn("concept:citylocatedinstate", pool[:2])

    def test_two_hop_composition_evidence_is_train_only_and_query_specific(self):
        memory = build_relation_memory_from_triples([
            ("concept_person_a", "concept:worksfor", "concept_organization_x"),
            ("concept_organization_x", "concept:hasofficeincity", "concept_city_q"),
            ("concept_person_a", "concept:personhasresidenceingeopoliticallocation", "concept_city_q"),
            # A held-out-looking pair with the same learned two-hop signature.
            ("concept_person_b", "concept:worksfor", "concept_organization_y"),
            ("concept_organization_y", "concept:hasofficeincity", "concept_city_r"),
        ])
        self.assertGreater(
            memory.evidence_score(
                "concept:personhasresidenceingeopoliticallocation",
                "concept_person_b", "concept_city_r",
            ),
            0.0,
        )
        self.assertEqual(
            memory.evidence_score(
                "concept:personhasresidenceingeopoliticallocation",
                "concept_person_b", "concept_city_unknown",
            ),
            0.0,
        )

    def test_teacher_forced_candidate_scoring_returns_one_score_per_candidate(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                # Include the wrapper and relation as deterministic token ids.
                ids = torch.tensor([[len(text) % 7 + 1, len(text) % 5 + 2]], dtype=torch.long)
                return {"input_ids": ids}

        class FakeModel:
            device = torch.device("cpu")

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask, **kwargs):
                # Uniform logits are sufficient to test alignment and batching.
                return SimpleNamespace(logits=torch.zeros(
                    input_ids.shape[0], input_ids.shape[1], 16
                ))

        scores = score_candidates(
            FakeModel(), FakeTokenizer(), torch.tensor([[1, 2, 3]]),
            ["concept:proxyfor", "concept:subpartof"], batch_size=2,
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in scores))
        per_candidate_scores = score_candidates(
            FakeModel(), FakeTokenizer(), [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
            ["concept:proxyfor", "concept:subpartof"], batch_size=1,
        )
        self.assertEqual(len(per_candidate_scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in per_candidate_scores))

    def test_counterfactual_score_rewards_supported_candidate_and_penalizes_rival(self):
        final, contrast = combine_cecr_scores(
            lm_scores=[-1.0, -0.9],
            prior_scores=[0.0, 0.0],
            evidence_scores=[1.0, 0.0],
            evidence_weight=0.2,
            counterfactual_weight=0.2,
        )
        self.assertEqual(contrast, [1.0, -1.0])
        self.assertGreater(final[0], final[1])

    def test_entity_type_fallback(self):
        self.assertEqual(infer_entity_type("concept_person_alice"), "person")
        self.assertEqual(infer_entity_type("concept_stateorprovince_x"), "geopolitical_location")


if __name__ == "__main__":
    unittest.main()
