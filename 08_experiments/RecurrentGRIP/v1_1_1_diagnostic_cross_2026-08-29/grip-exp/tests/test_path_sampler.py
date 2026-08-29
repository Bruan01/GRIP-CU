import unittest

from grip.tasks.recurrent_tasks.path_sampler import make_path_question


class PathSamplerTest(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "node_list": [
                "A has disabled access",
                "B does not have disabled access",
                "C has disabled access",
                "D does not have disabled access",
                "E has disabled access",
                "F does not have disabled access",
            ],
            "edge_index": [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5]],
        }

    def test_uses_bfs_shortest_distance_and_official_count_label(self):
        sample = make_path_question(
            graph_id="g0",
            local_index=7,
            graph=self.graph,
            question="How many stations are between A and D?",
            answer="2",
            question_type="StationShortestCount",
        )
        self.assertEqual(sample.true_hop, 3)
        self.assertEqual(sample.shortest_path, [0, 1, 2, 3])
        self.assertEqual(sample.frontiers, [[0], [1, 4], [2, 5], [3]])
        self.assertEqual(sample.answer, "2")

    def test_rejects_label_that_does_not_equal_distance_minus_one(self):
        with self.assertRaisesRegex(ValueError, "label/path mismatch"):
            make_path_question(
                graph_id="g0",
                local_index=0,
                graph=self.graph,
                question="How many stations are between A and F?",
                answer="2",
            )


if __name__ == "__main__":
    unittest.main()
