import random
import unittest

from recurrent_context_sampling import select_stratified_graph_context


class StratifiedGraphContextSamplerTest(unittest.TestCase):
    def _large_graph(self):
        nodes = [f"n{index}" for index in range(500)]
        edges = []
        for relation_index in range(198):
            relation = f"r{relation_index:03d}"
            edges.append([f"n{relation_index}", relation, f"n{relation_index + 1}"])
            edges.append([f"n{relation_index + 200}", relation, f"n{relation_index + 201}"])
        return {"node_list": nodes, "edge_list": edges}

    def test_relation_round_robin_preserves_edges_under_256_sample_budget(self):
        nodes, edges, manifest = select_stratified_graph_context(
            graph=self._large_graph(),
            node_quota=32,
            edge_quota=224,
            seed=2026,
        )

        self.assertEqual(len(nodes), 32)
        self.assertEqual(len(edges), 224)
        self.assertEqual(len(nodes) + len(edges), 256)
        self.assertEqual(manifest["selected_relation_count"], 198)
        self.assertEqual(manifest["relation_coverage"], 1.0)
        self.assertEqual(manifest["selected_node_count"], 32)
        self.assertEqual(manifest["selected_edge_count"], 224)

    def test_sampling_is_deterministic_without_changing_global_random_state(self):
        graph = self._large_graph()
        random.seed(91)
        expected_next = random.random()
        random.seed(91)

        first = select_stratified_graph_context(graph, 32, 224, seed=7)
        actual_next = random.random()
        second = select_stratified_graph_context(graph, 32, 224, seed=7)
        third = select_stratified_graph_context(graph, 32, 224, seed=8)

        self.assertEqual(actual_next, expected_next)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0:2], third[0:2])
        self.assertEqual(first[2]["selection_sha256"], second[2]["selection_sha256"])

    def test_deduplicates_edges_removes_self_loops_and_backfills_budget(self):
        graph = {
            "node_list": ["a", "b"],
            "edge_list": [
                ["a", "r1", "b"],
                ["a", "r1", "b"],
                ["b", "loop", "b"],
                ["b", "r2", "c"],
                ["c", "r3", "d"],
            ],
        }
        nodes, edges, manifest = select_stratified_graph_context(
            graph=graph,
            node_quota=1,
            edge_quota=1,
            seed=1,
        )

        self.assertEqual(len(nodes) + len(edges), 2)
        self.assertEqual(manifest["unique_edge_count"], 3)
        self.assertEqual(manifest["duplicate_edge_count"], 1)
        self.assertEqual(manifest["self_loop_count"], 1)
        self.assertTrue(all(source != target for source, _, target in edges))

    def test_train_qa_facts_are_anchored_before_relation_round_robin_fill(self):
        graph = {
            "node_list": ["a", "b", "c", "d", "e", "f"],
            "edge_list": [
                ["a", "r1", "b"],
                ["c", "r2", "d"],
                ["e", "r3", "f"],
            ],
        }
        train_samples = [
            {"split": "train", "answer": "r1", "source_node": "a", "target_node": "b"},
            {"split": "train", "answer": "r2", "source_node": "c", "target_node": "d"},
        ]
        _, edges, manifest = select_stratified_graph_context(
            graph=graph,
            node_quota=0,
            edge_quota=3,
            seed=2026,
            train_samples=train_samples,
        )

        self.assertIn(("a", "r1", "b"), edges)
        self.assertIn(("c", "r2", "d"), edges)
        self.assertEqual(manifest["train_qa_fact_count"], 2)
        self.assertEqual(manifest["train_qa_fact_covered_count"], 2)
        self.assertEqual(manifest["train_qa_fact_coverage"], 1.0)
        self.assertEqual(manifest["train_qa_entity_coverage"], 1.0)

    def test_manifest_reports_train_qa_relation_and_entity_coverage(self):
        graph = {
            "node_list": ["a", "b", "c", "d"],
            "edge_list": [["a", "r1", "b"], ["c", "r2", "d"]],
        }
        train_samples = [
            {"split": "train", "answer": "r1", "source_node": "a", "target_node": "b"},
            {"split": "train", "answer": "missing", "source_node": "x", "target_node": "y"},
            {"split": "test", "answer": "r2", "source_node": "c", "target_node": "d"},
        ]
        _, _, manifest = select_stratified_graph_context(
            graph=graph,
            node_quota=0,
            edge_quota=1,
            seed=4,
            train_samples=train_samples,
        )

        self.assertEqual(manifest["train_qa_count"], 2)
        self.assertEqual(manifest["train_qa_relation_covered_count"], 1)
        self.assertEqual(manifest["train_qa_relation_coverage"], 0.5)
        self.assertEqual(manifest["train_qa_entity_endpoint_count"], 4)
        self.assertEqual(manifest["train_qa_entity_covered_count"], 2)
        self.assertEqual(manifest["train_qa_entity_coverage"], 0.5)
        self.assertIn("selection_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
