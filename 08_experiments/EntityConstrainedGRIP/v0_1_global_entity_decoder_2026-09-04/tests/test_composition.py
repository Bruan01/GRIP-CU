import unittest

from entity_decoder.composition import build_train_compositions, annotate_composition, relation_frequency_bucket


class CompositionTest(unittest.TestCase):
    def test_exact_ordered_relation_sequence_defines_seen_composition(self):
        train = [{"path_relations": ["r1", "r2"]}, {"path_relations": ["r3"]}]
        known = build_train_compositions(train)
        self.assertEqual(annotate_composition({"path_relations": ["r1", "r2"]}, known), "seen-composition")
        self.assertEqual(annotate_composition({"path_relations": ["r2", "r1"]}, known), "novel-composition")

    def test_relation_frequency_uses_rarest_relation(self):
        counts = {"r1": 100, "r2": 3}
        self.assertEqual(relation_frequency_bucket(["r1", "r2"], counts, rare_max=5, frequent_min=21), "rare")
        self.assertEqual(relation_frequency_bucket(["r1"], counts, rare_max=5, frequent_min=21), "frequent")


if __name__ == "__main__":
    unittest.main()
