"""Automated tests for the required MLP and Transformer architectures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    MLPPolicy,
    TransformerPolicy,
    architecture_config,
    build_policy,
    greedy_action,
    masked_log_probabilities,
    masked_probabilities,
)


class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.observation = torch.rand(92, 5)
        self.mask = torch.ones(92, dtype=torch.bool)

    def test_output_shapes(self):
        for policy in [MLPPolicy(), TransformerPolicy()]:
            logits = policy(self.observation)
            self.assertEqual(logits.shape, (92,))
            self.assertTrue(torch.isfinite(logits).all())

    def test_mlp_architecture_is_exact(self):
        policy = MLPPolicy()
        linears = [module for module in policy.modules() if isinstance(module, nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linears],
            [(5, 64), (64, 64), (64, 1)],
        )
        self.assertEqual(sum(isinstance(m, nn.ReLU) for m in policy.modules()), 2)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in policy.modules()))
        self.assertFalse(any(isinstance(m, nn.Embedding) for m in policy.modules()))

    def test_transformer_configuration_is_exact(self):
        policy = TransformerPolicy()
        layer = policy.encoder_layer
        self.assertEqual(layer.self_attn.embed_dim, 64)
        self.assertEqual(layer.self_attn.num_heads, 4)
        self.assertEqual(layer.linear1.in_features, 64)
        self.assertEqual(layer.linear1.out_features, 128)
        self.assertEqual(layer.linear2.in_features, 128)
        self.assertEqual(layer.linear2.out_features, 64)
        self.assertFalse(layer.norm_first)  # post-layer normalization
        self.assertIs(layer.activation, torch.nn.functional.relu)
        self.assertEqual(layer.dropout.p, 0.0)
        self.assertEqual(layer.self_attn.dropout, 0.0)
        self.assertFalse(any(isinstance(m, nn.Embedding) for m in policy.modules()))
        # Exactly one encoder layer and no extra final normalization.
        self.assertFalse(hasattr(policy, "encoder"))
        self.assertFalse(hasattr(policy, "norm"))

    def test_all_parameters_are_trainable(self):
        for policy in [MLPPolicy(), TransformerPolicy()]:
            parameters = list(policy.parameters())
            self.assertGreater(len(parameters), 0)
            self.assertTrue(all(parameter.requires_grad for parameter in parameters))

    def test_forward_backward_gives_finite_gradients(self):
        for policy in [MLPPolicy(), TransformerPolicy()]:
            policy.zero_grad()
            logits = policy(self.observation)
            log_probs = masked_log_probabilities(logits, self.mask)
            loss = -log_probs[0]
            loss.backward()
            for parameter in policy.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_masked_actions_have_zero_probability(self):
        policy = MLPPolicy()
        logits = policy(self.observation)
        mask = torch.ones(92, dtype=torch.bool)
        mask[[3, 7, 11]] = False
        probabilities = masked_probabilities(logits, mask)
        self.assertTrue(torch.equal(probabilities[~mask], torch.zeros(3)))
        self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=6)
        log_probs = masked_log_probabilities(logits, mask)
        self.assertTrue(torch.isneginf(log_probs[~mask]).all())

    def test_greedy_ties_choose_smallest_id(self):
        logits = torch.zeros(92)
        mask = torch.ones(92, dtype=torch.bool)
        self.assertEqual(greedy_action(logits, mask), 0)
        mask[0] = False
        self.assertEqual(greedy_action(logits, mask), 1)
        logits[5] = 2.0
        logits[40] = 2.0
        mask = torch.zeros(92, dtype=torch.bool)
        mask[[5, 40]] = True
        self.assertEqual(greedy_action(logits, mask), 5)

    def test_selected_tokens_remain_visible_to_transformer_attention(self):
        policy = TransformerPolicy()
        baseline = policy(self.observation)
        changed = self.observation.clone()
        # Perturb one already-selected token (indicator was 0 at reset).
        changed[17, 2] = 1.0
        changed[17, 3] = 0.5
        changed[17, 4] = 0.0
        updated = policy(changed)
        self.assertGreater(float((updated - baseline).abs().max().detach()), 0.0)
        # A token's output therefore depends on other tokens, proving that the
        # network attends across participant positions rather than scoring each
        # row independently.
        self.assertGreater(float((updated[0] - baseline[0]).abs().detach()), 0.0)

    def test_build_policy_and_config(self):
        self.assertIsInstance(build_policy("reinforce_mlp"), MLPPolicy)
        self.assertIsInstance(build_policy("reinforce_transformer"), TransformerPolicy)
        mlp_config = architecture_config("reinforce_mlp")
        transformer_config = architecture_config("reinforce_transformer")
        self.assertEqual(mlp_config["hidden_sizes"], [64, 64])
        self.assertEqual(transformer_config["d_model"], 64)
        self.assertEqual(transformer_config["nhead"], 4)
        self.assertEqual(transformer_config["num_encoder_layers"], 1)
        self.assertEqual(transformer_config["dim_feedforward"], 128)
        self.assertEqual(transformer_config["dropout"], 0.0)
        self.assertFalse(transformer_config["norm_first"])
        with self.assertRaises(ValueError):
            build_policy("ppo")

    def test_bad_input_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            MLPPolicy()(torch.rand(92, 4))
        with self.assertRaises(ValueError):
            TransformerPolicy()(torch.rand(1, 92, 5))


if __name__ == "__main__":
    unittest.main()
