"""Regression: removing an edge must retain alignment of later edge text."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2] / "RecurrentGRIP/v1_1_nell23k_first_2026-08-29/grip-exp"
sys.path.insert(0, str(CODE))
from grip.tasks.utils import clean_graph


def test_clean_graph_keeps_original_edge_text_after_dropped_edges():
    edges = [["A", "self", "A"], ["A", "first", "B"],
             ["A", "duplicate", "B"], ["B", "later", "C"]]
    indices = np.array([[0, 0], [0, 1], [0, 1], [1, 2]])
    cleaned, cleaned_indices = clean_graph(edges, indices)
    assert cleaned == [edges[1], edges[3]]
    np.testing.assert_array_equal(cleaned_indices, [[0, 1], [1, 2]])
