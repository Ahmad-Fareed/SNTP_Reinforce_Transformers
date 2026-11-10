"""Automated tests for the fixed graph and network statistics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from network import (  # noqa: E402
    N_NODES,
    build_graph,
    compute_network_analysis,
    load_dataset,
)


class NetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.edges, cls.departments = load_dataset(ROOT / "data")
        cls.graph = build_graph(cls.nodes, cls.edges)
        cls.analysis = compute_network_analysis(cls.graph, cls.nodes, cls.departments)

    def test_participant_count_and_mapping(self):
        self.assertEqual(len(self.nodes), N_NODES)
        self.assertEqual(sorted(self.nodes["node_id"].tolist()), list(range(N_NODES)))
        self.assertTrue(self.nodes["original_id"].is_monotonic_increasing)
        self.assertEqual(sorted(self.departments["node_id"].tolist()), list(range(N_NODES)))

    def test_graph_is_undirected_unweighted_and_clean(self):
        self.assertEqual(self.graph.number_of_nodes(), 92)
        self.assertEqual(self.graph.number_of_edges(), 755)
        self.assertTrue(all(data == {} for _, _, data in self.graph.edges(data=True)))
        self.assertEqual(len(list(nx.isolates(self.graph))), 0)
        self.assertEqual(nx.number_connected_components(self.graph), 1)

    def test_density_matches_assignment_formula(self):
        expected = 2 * self.graph.number_of_edges() / (92 * 91)
        self.assertAlmostEqual(self.analysis["density"], expected, places=12)

    def test_degree_centrality_matches_degree_over_n_minus_one(self):
        for node in self.graph.nodes():
            expected = self.graph.degree(node) / 91
            self.assertAlmostEqual(
                self.analysis["degree_table"][node]["degree_centrality"], expected, places=12
            )

    def test_top10_sorted_by_degree_then_node_id(self):
        top10 = self.analysis["top10_degree_centrality"]
        self.assertEqual(len(top10), 10)
        keys = [(-row["degree"], row["node_id"]) for row in top10]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            [row["node_id"] for row in top10],
            sorted(self.graph.nodes(), key=lambda n: (-self.graph.degree(n), n))[:10],
        )


if __name__ == "__main__":
    unittest.main()
