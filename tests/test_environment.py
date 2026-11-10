"""Automated tests for the budgeted seed-selection environment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment import SeedSelectionEnv  # noqa: E402
from network import build_graph, load_dataset  # noqa: E402
from simulator import IndependentCascadeSimulator  # noqa: E402


class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, edges, _ = load_dataset(ROOT / "data")
        cls.graph = build_graph(nodes, edges)

    def make_env(self):
        return SeedSelectionEnv(self.graph)

    def test_reset_observation_shape_and_static_features(self):
        env = self.make_env()
        observation, mask = env.reset(900000)
        self.assertEqual(observation.shape, (92, 5))
        self.assertEqual(mask.shape, (92,))
        self.assertTrue(mask.all())
        np.testing.assert_allclose(
            observation[:, 0], [self.graph.degree(node) / 91 for node in range(92)]
        )
        np.testing.assert_allclose(
            observation[:, 1], [__import__("networkx").clustering(self.graph, node) for node in range(92)]
        )
        np.testing.assert_array_equal(observation[:, 2], np.zeros(92))
        np.testing.assert_array_equal(observation[:, 3], np.zeros(92))
        np.testing.assert_allclose(observation[:, 4], np.ones(92))

    def test_dynamic_features_and_mask_update(self):
        env = self.make_env()
        observation, _ = env.reset(900000)
        observation, reward, terminated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertEqual(observation[0, 2], 1.0)
        self.assertEqual(observation[0, 3], 0.0)
        self.assertFalse(env.action_mask[0])
        self.assertEqual(int(env.action_mask.sum()), 91)
        np.testing.assert_allclose(observation[:, 4], np.full(92, 0.8))
        # Every neighbour of the selected node sees the updated fraction.
        for neighbour in self.graph.neighbors(0):
            self.assertAlmostEqual(
                observation[neighbour, 3],
                1.0 / self.graph.degree(neighbour),
                places=6,
            )
        self.assertIn("selected_nodes", info)

    def test_duplicate_and_invalid_actions_are_rejected(self):
        env = self.make_env()
        env.reset(900000)
        env.step(0)
        with self.assertRaises(ValueError):
            env.step(0)
        with self.assertRaises(ValueError):
            env.step(-1)
        with self.assertRaises(ValueError):
            env.step(92)
        with self.assertRaises(ValueError):
            env.step(3.5)
        with self.assertRaises(ValueError):
            env.step(True)

    def test_diffusion_runs_only_after_fifth_action(self):
        calls = []

        def factory(*args, **kwargs):
            calls.append(1)
            return IndependentCascadeSimulator(*args, **kwargs)

        with patch("environment.IndependentCascadeSimulator", side_effect=factory):
            env = self.make_env()
            env.reset(900000)
            for action in [0, 1, 2, 3]:
                _, reward, terminated, info = env.step(action)
                self.assertEqual(reward, 0.0)
                self.assertFalse(terminated)
                self.assertNotIn("final_spread", info)
            self.assertEqual(calls, [])
            observation, reward, terminated, info = env.step(4)
            self.assertEqual(calls, [1])
        self.assertTrue(terminated)
        self.assertGreaterEqual(info["final_spread"], 5)
        self.assertAlmostEqual(reward, info["final_spread"] / 92)
        self.assertEqual(len(set(env.selected_nodes)), 5)
        self.assertEqual(int(env.action_mask.sum()), 87)

    def test_reward_equals_spread_over_92(self):
        env = self.make_env()
        env.reset(900000)
        for action in [84, 58, 14, 12]:
            _, reward, terminated, _ = env.step(action)
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
        _, reward, terminated, info = env.step(27)
        self.assertTrue(terminated)
        self.assertEqual(info["final_spread"], 62)
        self.assertAlmostEqual(reward, 62 / 92, places=12)

    def test_reset_reproduces_same_episode(self):
        actions = [84, 58, 14, 12, 27]
        env = self.make_env()
        rewards = []
        for _ in range(2):
            env.reset(900000)
            for action in actions:
                _, reward, terminated, info = env.step(action)
            rewards.append((reward, info["final_spread"]))
        self.assertEqual(rewards[0], rewards[1])

    def test_episode_terminates_after_fifth_selection(self):
        env = self.make_env()
        env.reset(900000)
        for action in [0, 1, 2, 3, 4]:
            _, _, terminated, _ = env.step(action)
        self.assertTrue(terminated)
        with self.assertRaises(RuntimeError):
            env.step(5)
        # Reset is still possible after termination.
        observation, mask = env.reset(900001)
        self.assertEqual(observation.shape, (92, 5))
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
