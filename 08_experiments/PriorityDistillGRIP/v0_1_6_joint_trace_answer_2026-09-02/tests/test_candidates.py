import unittest

from priority_distill.candidates import build_candidate_pools, relation_overlap


def make_row(index, relations, answer=None, split="train"):
    depth = len(relations)
    nodes = [f"n{index}_{offset}" for offset in range(depth + 1)]
    return {
        "task_id": f"{split}:{index}",
        "text": f"question {index}",
        "answer": answer or nodes[-1],
        "depth_label": depth,
        "path_nodes": nodes,
        "path_relations": relations,
        "path_edges": [
            {"head": nodes[offset], "relation": relation, "tail": nodes[offset + 1]}
            for offset, relation in enumerate(relations)
        ],
        "split": split,
        "depth_source": "exact_directed_shortest_path",
        "shortest_path_verified": True,
        "unique_relation_chain_answer": True,
    }


class CandidateTests(unittest.TestCase):
    def test_relation_overlap_rewards_shared_relation_positions(self):
        self.assertGreater(relation_overlap(["r1", "r2"], ["r1", "r3"]), relation_overlap(["r1", "r2"], ["x", "y"]))

    def test_pool_has_one_gold_and_three_train_only_distractors(self):
        rows = [
            make_row(0, ["r1", "r2"], answer="gold"),
            make_row(1, ["r1", "r3"]),
            make_row(2, ["r1", "r4"]),
            make_row(3, ["x", "r2"]),
            make_row(4, ["x", "y"]),
        ]
        pools = build_candidate_pools(rows, distractor_count=3, seed=7)
        pool = pools["train:0"]
        self.assertEqual(sum(candidate["is_gold"] for candidate in pool), 1)
        self.assertEqual(len(pool), 4)
        self.assertTrue(all(candidate["source_split"] == "train" for candidate in pool))
        distractor_answers = [candidate["answer"] for candidate in pool if not candidate["is_gold"]]
        self.assertNotEqual(set(distractor_answers), {"gold"})
        self.assertEqual(len(distractor_answers), len(set(distractor_answers)))

    def test_pool_is_deterministic(self):
        rows = [make_row(i, ["r1", f"r{i}"]) for i in range(6)]
        self.assertEqual(
            build_candidate_pools(rows, distractor_count=3, seed=11),
            build_candidate_pools(list(reversed(rows)), distractor_count=3, seed=11),
        )


if __name__ == "__main__":
    unittest.main()
