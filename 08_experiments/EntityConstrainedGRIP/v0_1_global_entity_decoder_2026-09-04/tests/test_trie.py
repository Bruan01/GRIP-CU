import unittest

from entity_decoder.trie import EntityTokenTrie, TrieConstraint


class TrieTest(unittest.TestCase):
    def setUp(self):
        self.trie = EntityTokenTrie({"alpha": [10, 11], "alpine": [10, 12], "beta": [20]})

    def test_returns_valid_next_tokens_and_eos_only_at_terminal(self):
        self.assertEqual(self.trie.allowed_next([]), {10, 20})
        self.assertEqual(self.trie.allowed_next([10]), {11, 12})
        self.assertEqual(self.trie.allowed_next([20]), set())
        self.assertTrue(self.trie.is_terminal([20]))

    def test_maps_complete_token_sequence_to_unique_entity(self):
        self.assertEqual(self.trie.entity_for([10, 12]), "alpine")
        with self.assertRaises(KeyError):
            self.trie.entity_for([10])

    def test_constraint_uses_generation_offset_and_forces_eos(self):
        constraint = TrieConstraint(self.trie, prompt_length=3, eos_token_id=2)
        self.assertEqual(constraint.allowed_tokens(0, [5, 6, 7]), [10, 20])
        self.assertEqual(constraint.allowed_tokens(0, [5, 6, 7, 20]), [2])

    def test_terminal_prefix_allows_eos_and_longer_entities(self):
        trie = EntityTokenTrie({"short": [1], "long": [1, 2]})
        constraint = TrieConstraint(trie, prompt_length=1, eos_token_id=9)
        self.assertEqual(constraint.allowed_tokens(0, [7, 1]), [2, 9])

    def test_rejects_tokenization_collisions(self):
        with self.assertRaises(ValueError):
            EntityTokenTrie({"one": [1, 2], "two": [1, 2]})


if __name__ == "__main__":
    unittest.main()
