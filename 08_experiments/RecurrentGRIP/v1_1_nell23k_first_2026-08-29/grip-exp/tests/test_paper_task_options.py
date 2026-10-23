"""Strict-paper task-generation options (GRIP Sec. 4.2-4.3, App. A.1)."""

import random
import unittest

import numpy as np

from grip.tasks.train_tasks.gen_context import GenGraphContextTask
from grip.tasks.train_tasks.gen_reasoning_qa import GenReasoningQATask
from grip.tasks.utils import clean_graph, dedup_triples, sample_indices, sample_rooted_subgraph_edges


class _ChatTokenizer:
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class _RecordingGenerator:
    batch_size = 1

    def __init__(self):
        self.prompts = []

    def inference(self, prompts, system_prompt):
        self.prompts.extend(prompts)
        return [{"response": "Changed: no\nQuestion: q?\nAnswer: a"} for _ in prompts]


def _graph():
    # A -r1-> B, A -r2-> B (parallel), B -r3-> C, C -r4-> D, D -r5-> E; F is not in any edge.
    return {
        "node_list": ["A", "B", "C", "D", "E", "F"],
        "edge_list": [["A", "r1", "B"], ["A", "r2", "B"], ["B", "r3", "C"],
                      ["C", "r4", "D"], ["D", "r5", "E"], ["A", "r1", "B"]],
        "edge_index": [[0, 1], [0, 1], [1, 2], [2, 3], [3, 4], [0, 1]],
    }


class GraphUtilsTest(unittest.TestCase):
    def test_dedup_triples_keeps_parallel_relations(self):
        graph = _graph()
        edges, index = dedup_triples(graph["edge_list"], np.array(graph["edge_index"]))
        self.assertEqual([e[1] for e in edges], ["r1", "r2", "r3", "r4", "r5"])
        self.assertEqual(index.shape, (5, 2))
        collapsed, _ = clean_graph(graph["edge_list"], np.array(graph["edge_index"]))
        self.assertNotIn("r2", [e[1] for e in collapsed])

    def test_rooted_subgraph_includes_edge_zero_and_is_rooted(self):
        np.random.seed(0)
        edge_index = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        positions = sample_rooted_subgraph_edges(edge_index, 5, root=0, hop=3, max_nodes_per_hop=3)
        self.assertEqual(positions, [0, 1, 2])  # 3 hops from A: A-B, B-C, C-D

    def test_rooted_subgraph_caps_neighbors_per_hop(self):
        np.random.seed(1)
        star = np.array([[0, i] for i in range(1, 21)])
        positions = sample_rooted_subgraph_edges(star, 21, root=0, hop=3, max_nodes_per_hop=3)
        self.assertTrue(1 <= len(positions) <= 3)
        self.assertEqual(len(set(positions)), len(positions))

    def test_uniform_sampling_ignores_weights(self):
        np.random.seed(0)
        draws = sample_indices(4, np.array([1.0, 0.0, 0.0, 0.0]), 400, uniform=True)
        self.assertEqual(set(draws), {0, 1, 2, 3})
        weighted = sample_indices(4, np.array([1.0, 0.0, 0.0, 0.0]), 50, uniform=False)
        self.assertEqual(set(weighted), {0})


class ContextTaskTest(unittest.TestCase):
    def _run(self, **kwargs):
        np.random.seed(0)
        task = GenGraphContextTask(
            graph_list=[_graph()], title_list=["g"], tokenizer=_ChatTokenizer(),
            context_upsampling=False, format_as_instruction=False, **kwargs,
        )
        end = _ChatTokenizer.eos_token + "\n"
        return [s[: -len(end)] if s.endswith(end) else s for s in task()[0]]

    def test_official_default_collapses_pairs_and_recites_all_nodes(self):
        samples = self._run()
        self.assertFalse(any(" r2 " in s for s in samples))
        self.assertIn("In context graph g, there is a node F.", samples)

    def test_paper_options_keep_every_triple_and_train_entities(self):
        samples = self._run(keep_parallel_edges=True, train_entity_nodes_only=True)
        edges = [s for s in samples if " is " in s and "there is a node" not in s]
        nodes = [s for s in samples if "there is a node" in s]
        self.assertEqual(len(nodes), 5)  # A-E; F has no train edge
        self.assertEqual(sorted(s.split(" is ")[1].split()[0] for s in edges), ["r1", "r2", "r3", "r4", "r5"])
        self.assertNotIn("In context graph g, there is a node F.", samples)
        self.assertTrue(all(not s.startswith("system:") for s in samples))  # plain text


class ReasoningTaskTest(unittest.TestCase):
    def _prompts(self, **kwargs):
        random.seed(0)
        np.random.seed(0)
        generator = _RecordingGenerator()
        refer = [{"questions": ["What is the relation between x and y?"] * 8, "answers": ["r"] * 8}]
        task = GenReasoningQATask(
            graph_list=[_graph()], title_list=["g"], tokenizer=_ChatTokenizer(),
            task_generator=generator, refer_data=refer, num_reason_qa=40, **kwargs,
        )
        task()
        return [p for p in generator.prompts if "Generate two" in p or "Create two" in p]

    def test_rooted_random_type_prompts(self):
        prompts = self._prompts(rooted_reasoning_subgraph=True, random_reasoning_type=True, uniform_sampling=True)
        self.assertEqual(len(prompts), 40)
        markers = ("multi-hop", "full subgraph", "answered yes", "ANOTHER graph")
        kinds = [next(m for m in markers if m in p) for p in prompts]
        self.assertEqual(set(kinds), set(markers))
        round_robin = [markers[i % 4] for i in (1, 0, 2, 3)] * 10
        self.assertNotEqual(kinds, round_robin)
        for prompt in prompts:
            # rooted subgraph edges come from a connected chain, never empty
            self.assertIn(" is r", prompt)

    def test_default_round_robin_is_unchanged(self):
        prompts = self._prompts()
        self.assertEqual(len(prompts), 40)
        self.assertIn("ANOTHER graph", prompts[3])
        self.assertIn("ANOTHER graph", prompts[7])


if __name__ == "__main__":
    unittest.main()
