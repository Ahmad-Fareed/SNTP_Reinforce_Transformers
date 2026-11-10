"""Automated tests for the REINFORCE training procedure."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from network import build_graph, load_dataset  # noqa: E402
from reinforce import (  # noqa: E402
    EPISODES_PER_RUN,
    EPISODES_PER_UPDATE,
    UPDATE_COUNT,
    load_checkpoint,
    save_training_diagnostics,
    train_policy,
)


class ReinforceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temp_dir.name)
        nodes, edges, _ = load_dataset(ROOT / "data")
        cls.graph = build_graph(nodes, edges)
        cls.result = train_policy(
            "reinforce_mlp",
            cls.graph,
            train_seed=7,
            output_dir=cls.output_dir,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_training_record_count_and_columns(self):
        records = self.result["records"]
        self.assertEqual(len(records), EPISODES_PER_RUN)
        self.assertEqual(
            list(records[0].keys()),
            ["method", "train_seed", "episode", "final_spread", "total_reward"],
        )
        self.assertEqual([row["episode"] for row in records], list(range(EPISODES_PER_RUN)))
        self.assertTrue(all(row["method"] == "reinforce_mlp" for row in records))
        self.assertTrue(all(row["train_seed"] == 7 for row in records))

    def test_rewards_are_spread_over_92(self):
        for row in self.result["records"]:
            self.assertAlmostEqual(row["total_reward"], row["final_spread"] / 92, places=12)
            self.assertGreaterEqual(row["final_spread"], 5)
            self.assertLessEqual(row["final_spread"], 92)

    def test_exactly_fifty_updates_of_twenty_episodes(self):
        diagnostics = self.result["diagnostics"]
        self.assertEqual(diagnostics["optimizer_steps"], UPDATE_COUNT)
        self.assertEqual(diagnostics["optimizer_steps"], 50)
        self.assertEqual(len(diagnostics["batch_episode_counts"]), 50)
        self.assertTrue(all(count == EPISODES_PER_UPDATE for count in diagnostics["batch_episode_counts"]))
        self.assertEqual(len(diagnostics["losses"]), 50)

    def test_diffusion_seed_formula(self):
        expected = [1_000_000 + 10_000 * 7 + episode for episode in range(1000)]
        self.assertEqual(self.result["diagnostics"]["diffusion_seeds"], expected)

    def test_losses_and_gradients_are_finite(self):
        diagnostics = self.result["diagnostics"]
        self.assertTrue(diagnostics["finite_losses"])
        self.assertTrue(diagnostics["finite_gradients"])
        self.assertTrue(all(math.isfinite(value) for value in diagnostics["losses"]))
        self.assertTrue(all(math.isfinite(value) for value in diagnostics["gradient_norms"]))

    def test_gradients_are_clipped_to_one(self):
        diagnostics = self.result["diagnostics"]
        self.assertTrue(diagnostics["gradients_clipped_to_max"])
        self.assertLessEqual(diagnostics["max_gradient_norm_after_clip"], 1.0 + 1e-6)

    def test_baseline_moves_after_update_with_prescribed_rule(self):
        diagnostics = self.result["diagnostics"]
        self.assertTrue(diagnostics["baseline_sequence_ok"])
        baseline = 0.0
        for batch_mean in diagnostics["batch_mean_rewards"]:
            baseline = 0.9 * baseline + 0.1 * batch_mean
        self.assertAlmostEqual(baseline, diagnostics["final_baseline"], places=12)
        self.assertEqual(diagnostics["baselines_before_update"][0], 0.0)

    def test_checkpoint_contains_parameters_and_architecture(self):
        path = self.output_dir / "checkpoints" / "reinforce_mlp_seed7.pt"
        self.assertTrue(path.exists())
        policy, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["method"], "reinforce_mlp")
        self.assertEqual(checkpoint["train_seed"], 7)
        self.assertEqual(checkpoint["architecture"]["hidden_sizes"], [64, 64])
        self.assertEqual(checkpoint["training_settings"]["optimizer"], "Adam")
        self.assertAlmostEqual(checkpoint["training_settings"]["learning_rate"], 0.001)
        trained_parameters = [p.detach().clone() for p in self.result["policy"].parameters()]
        loaded_parameters = [p.detach().clone() for p in policy.parameters()]
        self.assertEqual(len(trained_parameters), len(loaded_parameters))
        for trained, loaded in zip(trained_parameters, loaded_parameters):
            self.assertTrue(torch.equal(trained, loaded))

    def test_diagnostics_can_be_saved(self):
        path = self.output_dir / "training_diagnostics.json"
        save_training_diagnostics([self.result], path)
        self.assertTrue(path.exists())
        self.assertNotIn("policy", path.read_text(encoding="utf-8"))

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            train_policy("ppo", self.graph, 7, self.output_dir, verbose=False)
        with self.assertRaises(ValueError):
            train_policy("reinforce_mlp", self.graph, 8, self.output_dir, verbose=False)


if __name__ == "__main__":
    unittest.main()
