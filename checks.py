"""Automated acceptance checks written to results/checks.json.

Every check executes real code.  A failed expectation is recorded as
``passed: false`` with the exception message, and ``main.py`` stops after
writing the file, so a broken implementation cannot silently look successful.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch

from environment import SeedSelectionEnv
from models import (
    TransformerPolicy,
    MLPPolicy,
    greedy_action,
    masked_probabilities,
)
from network import N_NODES
from reinforce import TRAINING_SEEDS
from simulator import IndependentCascadeSimulator

CHECK_NAMES = [
    "empty_seed_diffusion",
    "probability_zero_diffusion",
    "probability_one_diffusion",
    "diffusion_reproducibility",
    "duplicate_action_rejection",
    "five_distinct_selected_participants",
    "correct_reward_calculation",
    "zero_probability_for_masked_actions",
    "finite_training_losses",
    "finite_training_gradients",
]


def _run_check(results: dict[str, Any], name: str, function: Callable[[], str]) -> None:
    try:
        details = function()
        results[name] = {"passed": True, "details": details}
    except Exception as exc:  # keep the failure visible instead of aborting the file
        results[name] = {"passed": False, "details": f"{type(exc).__name__}: {exc}"}


def run_checks(
    graph: nx.Graph,
    training_diagnostics: list[dict[str, Any]],
    checkpoint_dir: Path,
    evaluation,
    output_path: Path,
) -> dict[str, Any]:
    """Execute all checks, write JSON, and raise if anything failed."""
    results: dict[str, Any] = {}
    fixed_selection = [84, 58, 14, 12, 27]

    def empty_seed() -> str:
        spreads = {
            seed: IndependentCascadeSimulator(graph, seed).simulate([])
            for seed in (1, 42, 900000)
        }
        assert all(value == 0 for value in spreads.values()), spreads
        return f"empty seeds gave zero spread for diffusion seeds {sorted(spreads)}"

    def probability_zero() -> str:
        simulator = IndependentCascadeSimulator(graph, 7, probability=0.0)
        seeds = [0, 5, 10, 20, 30]
        spread = simulator.simulate(seeds)
        assert spread == len(seeds), spread
        assert simulator.simulate([3]) == 1
        return f"p=0 with seeds {seeds} produced {spread} active nodes"

    def probability_one() -> str:
        simulator = IndependentCascadeSimulator(graph, 7, probability=1.0)
        expected = len(nx.node_connected_component(graph, 0))
        spread = simulator.simulate([0, 5, 10, 20, 30])
        assert expected == N_NODES, expected
        assert spread == expected, (spread, expected)
        return f"p=1 reached all {spread} participants in the connected graph"

    def reproducibility() -> str:
        spreads = {}
        for diffusion_seed in (42, 900000, 1_234_567):
            first = IndependentCascadeSimulator(graph, diffusion_seed).simulate(fixed_selection)
            second = IndependentCascadeSimulator(graph, diffusion_seed).simulate(fixed_selection)
            assert first == second, (diffusion_seed, first, second)
            spreads[diffusion_seed] = first
        return f"identical spreads on repeat runs: {spreads}"

    def duplicate_rejection() -> str:
        env = SeedSelectionEnv(graph)
        env.reset(900000)
        env.step(0)
        try:
            env.step(0)
        except ValueError as exc:
            assert env.selected_nodes == [0]
            return f"second selection of node 0 raised: {exc}"
        raise AssertionError("duplicate action was not rejected")

    def five_distinct() -> str:
        env = SeedSelectionEnv(graph)
        env.reset(900000)
        terminated = False
        for action in fixed_selection:
            _, _, terminated, _ = env.step(action)
        assert terminated
        selected = env.selected_nodes
        assert len(selected) == 5 and len(set(selected)) == 5, selected
        assert int(env.action_mask.sum()) == N_NODES - 5
        return f"five distinct selections {selected}; {int(env.action_mask.sum())} actions left"

    def reward_calculation() -> str:
        env = SeedSelectionEnv(graph)
        env.reset(900000)
        early_rewards = []
        for action in fixed_selection[:4]:
            _, reward, terminated, _ = env.step(action)
            early_rewards.append(reward)
            assert not terminated
        assert early_rewards == [0.0, 0.0, 0.0, 0.0], early_rewards
        _, reward, terminated, info = env.step(fixed_selection[4])
        assert terminated
        expected_spread = IndependentCascadeSimulator(graph, 900000).simulate(fixed_selection)
        assert info["final_spread"] == expected_spread, (info, expected_spread)
        assert abs(reward - expected_spread / N_NODES) < 1e-12
        return (
            f"first four rewards were zero; fifth reward {reward:.12f} = "
            f"{expected_spread}/{N_NODES}"
        )

    def masked_probability() -> str:
        observation = torch.zeros(N_NODES, 5)
        mask = torch.ones(N_NODES, dtype=torch.bool)
        mask[[0, 5, 84]] = False
        for policy in (MLPPolicy(), TransformerPolicy()):
            policy.eval()
            with torch.no_grad():
                logits = policy(observation)
                probabilities = masked_probabilities(logits, mask)
            assert torch.equal(probabilities[~mask], torch.zeros(3)), probabilities[~mask]
            assert abs(float(probabilities.sum().detach()) - 1.0) < 1e-6
            assert greedy_action(logits, mask) not in {0, 5, 84}
        return "MLP and Transformer gave exactly zero probability to masked actions"

    def finite_losses() -> str:
        assert len(training_diagnostics) == 6, len(training_diagnostics)
        values = []
        for diagnostics in training_diagnostics:
            losses = np.asarray(diagnostics["losses"], dtype=float)
            assert len(losses) == 50, len(losses)
            assert np.isfinite(losses).all()
            assert diagnostics["finite_losses"]
            values.append((diagnostics["method"], diagnostics["train_seed"], float(losses[-1])))
        return f"50 finite losses in each of 6 runs; final losses={values}"

    def finite_gradients() -> str:
        assert len(training_diagnostics) == 6, len(training_diagnostics)
        norms = []
        for diagnostics in training_diagnostics:
            gradient_norms = np.asarray(diagnostics["gradient_norms"], dtype=float)
            assert len(gradient_norms) == 50, len(gradient_norms)
            assert np.isfinite(gradient_norms).all()
            assert diagnostics["finite_gradients"]
            assert diagnostics["gradients_clipped_to_max"]
            norms.append(float(gradient_norms.max()))
        return f"all gradients finite; maximum pre-clip norms per run={norms}"

    _run_check(results, "empty_seed_diffusion", empty_seed)
    _run_check(results, "probability_zero_diffusion", probability_zero)
    _run_check(results, "probability_one_diffusion", probability_one)
    _run_check(results, "diffusion_reproducibility", reproducibility)
    _run_check(results, "duplicate_action_rejection", duplicate_rejection)
    _run_check(results, "five_distinct_selected_participants", five_distinct)
    _run_check(results, "correct_reward_calculation", reward_calculation)
    _run_check(results, "zero_probability_for_masked_actions", masked_probability)
    _run_check(results, "finite_training_losses", finite_losses)
    _run_check(results, "finite_training_gradients", finite_gradients)

    # Additional protocol checks are useful evidence, not substitutes for the
    # ten required checks above.
    def training_protocol() -> str:
        assert len(training_diagnostics) == 6
        for diagnostics in training_diagnostics:
            assert diagnostics["episodes"] == 1000
            assert diagnostics["optimizer_steps"] == 50
            assert all(count == 20 for count in diagnostics["batch_episode_counts"])
            assert diagnostics["baseline_sequence_ok"]
            assert diagnostics["diffusion_seeds"] == [
                1_000_000 + 10_000 * diagnostics["train_seed"] + episode
                for episode in range(1000)
            ]
        return "6 runs x (1,000 episodes, 50 updates of 20 episodes, correct seeds and baseline)"

    def checkpoint_protocol() -> str:
        expected = {
            (method, seed)
            for method in ("reinforce_mlp", "reinforce_transformer")
            for seed in TRAINING_SEEDS
        }
        found = set()
        for method, seed in expected:
            path = Path(checkpoint_dir) / f"{method}_seed{seed}.pt"
            assert path.exists(), path
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert checkpoint["method"] == method
            assert checkpoint["train_seed"] == seed
            assert checkpoint["architecture"]["type"] in {"mlp", "transformer"}
            assert len(checkpoint["state_dict"]) > 0
            found.add((method, seed))
        assert found == expected
        return f"loaded all {len(expected)} final checkpoints with architecture metadata"

    def evaluation_protocol() -> str:
        assert len(evaluation) == 800, len(evaluation)
        expected_seeds = set(range(900000, 900100))
        assert set(evaluation["diffusion_seed"].unique()) == expected_seeds
        combinations = {
            ("random", -1),
            ("degree", -1),
            *{("reinforce_mlp", seed) for seed in TRAINING_SEEDS},
            *{("reinforce_transformer", seed) for seed in TRAINING_SEEDS},
        }
        assert combinations == set(evaluation[["method", "train_seed"]].itertuples(index=False, name=None))
        for (method, train_seed), subset in evaluation.groupby(["method", "train_seed"]):
            assert len(subset) == 100, (method, train_seed, len(subset))
            assert set(subset["diffusion_seed"]) == expected_seeds
            for value in subset["selected_nodes"]:
                selected = json.loads(value)
                assert len(selected) == 5 and len(set(selected)) == 5, selected
                assert all(0 <= node < N_NODES for node in selected)
            assert np.allclose(
                subset["total_reward"].to_numpy(),
                subset["final_spread"].to_numpy() / N_NODES,
            )
            assert (subset["selection_seconds"] >= 0).all()
        return "800 episodes = 8 method/seed groups x 100 common diffusion seeds"

    _run_check(results, "training_protocol", training_protocol)
    _run_check(results, "checkpoint_protocol", checkpoint_protocol)
    _run_check(results, "evaluation_protocol", evaluation_protocol)

    all_passed = all(entry["passed"] for entry in results.values())
    payload = {
        "all_passed": all_passed,
        "required_checks": CHECK_NAMES,
        "checks": results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if not all_passed:
        failed = [name for name, entry in results.items() if not entry["passed"]]
        raise RuntimeError(f"acceptance checks failed: {failed}")
    return payload
