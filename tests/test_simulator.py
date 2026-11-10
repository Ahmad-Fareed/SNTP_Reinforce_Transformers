"""Automated tests for the reproducibile Independent Cascade simulator."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from network import build_graph, load_dataset  # noqa: E402
from simulator import IndependentCascadeSimulator  # noqa: E402


class SimulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, edges, _ = load_dataset(ROOT / "data")
        cls.graph = build_graph(nodes, edges)
        cls.seeds = [0, 10, 20, 30, 40]

    def test_arc_list_is_symmetric_and_sorted(self):
        simulator = IndependentCascadeSimulator(self.graph, 1, probability=0.10)
        arcs = simulator._arcs
        self.assertEqual(len(arcs), 2 * self.graph.number_of_edges())
        self.assertEqual(arcs, sorted(arcs))
        self.assertEqual(set(arcs), {(u, v) for u, v in self.graph.edges()} | {
            (v, u) for u, v in self.graph.edges()
        })

    def test_one_draw_per_arc_in_sorted_order(self):
        simulator = IndependentCascadeSimulator(self.graph, 12345, probability=0.10)
        expected = np.random.default_rng(12345).random(simulator.n_arcs)
        np.testing.assert_array_equal(simulator._draws, expected)
        np.testing.assert_array_equal(simulator._live_arcs, expected < 0.10)

    def test_empty_seed_set_gives_zero(self):
        simulator = IndependentCascadeSimulator(self.graph, 1, probability=0.10)
        self.assertEqual(simulator.simulate([]), 0)

    def test_probability_zero_returns_only_distinct_seeds(self):
        simulator = IndependentCascadeSimulator(self.graph, 7, probability=0.0)
        self.assertEqual(simulator.simulate(self.seeds), len(set(self.seeds)))
        self.assertEqual(simulator.simulate([3]), 1)

    def test_probability_one_reaches_whole_component(self):
        simulator = IndependentCascadeSimulator(self.graph, 7, probability=1.0)
        expected = len(nx.node_connected_component(self.graph, self.seeds[0]))
        self.assertEqual(simulator.simulate(self.seeds), expected)
        self.assertEqual(expected, 92)

    def test_same_inputs_give_identical_spread(self):
        first = IndependentCascadeSimulator(self.graph, 900000, probability=0.10)
        second = IndependentCascadeSimulator(self.graph, 900000, probability=0.10)
        self.assertEqual(first.simulate(self.seeds), second.simulate(self.seeds))
        np.testing.assert_array_equal(first._draws, second._draws)

    def test_different_realisations_are_reproducible_on_their_own(self):
        first = IndependentCascadeSimulator(self.graph, 900000, probability=0.10)
        other = IndependentCascadeSimulator(self.graph, 900001, probability=0.10)
        repeat = IndependentCascadeSimulator(self.graph, 900000, probability=0.10)
        self.assertEqual(first.simulate(self.seeds), repeat.simulate(self.seeds))
        self.assertFalse(np.array_equal(first._draws, other._draws))

    def test_invalid_inputs_raise(self):
        simulator = IndependentCascadeSimulator(self.graph, 1, probability=0.10)
        with self.assertRaises(ValueError):
            simulator.simulate([0, 0])
        with self.assertRaises(ValueError):
            simulator.simulate([92])
        with self.assertRaises(ValueError):
            IndependentCascadeSimulator(self.graph, 1, probability=1.1)

    def test_random_state_is_private(self):
        simulator = IndependentCascadeSimulator(self.graph, 1, probability=0.10)
        public = {name for name in dir(simulator) if not name.startswith("_")}
        self.assertEqual(public, {"n_arcs", "simulate"})


if __name__ == "__main__":
    unittest.main()
