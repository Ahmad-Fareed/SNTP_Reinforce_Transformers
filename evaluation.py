"""Controlled evaluation of Random, Degree, and the six learned checkpoints.

All methods use diffusion seeds 900000-900099.  Neural policies act greedily
with gradients disabled.  Selection time is measured around the decision
mechanism only; environment updates and diffusion are excluded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from environment import SeedSelectionEnv
from models import greedy_action
from reinforce import TRAINING_SEEDS, load_checkpoint

EVALUATION_SEEDS = list(range(900000, 900100))
LEARNED_METHODS = ("reinforce_mlp", "reinforce_transformer")
SUMMARY_METHODS = ("random", "degree", "reinforce_mlp", "reinforce_transformer")
EVALUATION_COLUMNS = [
    "method",
    "train_seed",
    "diffusion_seed",
    "selected_nodes",
    "final_spread",
    "total_reward",
    "selection_seconds",
]
SUMMARY_COLUMNS = [
    "method",
    "train_seed",
    "spread_mean",
    "spread_std",
    "reward_mean",
    "selection_seconds_mean",
    "training_seconds",
]


def degree_selection(graph) -> list[int]:
    """Five highest-degree nodes; ties resolved by the smaller node ID."""
    return sorted(graph.nodes(), key=lambda node: (-graph.degree(node), node))[:5]


def _finish_episode(
    env: SeedSelectionEnv, actions: list[int], diffusion_seed: int
) -> dict[str, Any]:
    env.reset(diffusion_seed)
    reward = 0.0
    info: dict[str, Any] = {}
    terminated = False
    for action in actions:
        _, reward, terminated, info = env.step(action)
    if not terminated or len(actions) != 5 or len(set(actions)) != 5:
        raise RuntimeError("evaluation episode did not select five distinct participants")
    return {
        "selected_nodes": [int(action) for action in actions],
        "final_spread": int(info["final_spread"]),
        "total_reward": float(reward),
    }


def evaluate_random(graph, seeds: list[int] = EVALUATION_SEEDS) -> list[dict[str, Any]]:
    """Uniform five-node sample from a generator seeded with diffusion_seed + 1e7."""
    env = SeedSelectionEnv(graph)
    rows = []
    for diffusion_seed in seeds:
        selection_start = time.perf_counter()
        generator = np.random.default_rng(diffusion_seed + 10_000_000)
        actions = [int(node) for node in generator.choice(92, size=5, replace=False)]
        selection_seconds = time.perf_counter() - selection_start
        episode = _finish_episode(env, actions, diffusion_seed)
        rows.append(
            {
                "method": "random",
                "train_seed": -1,
                "diffusion_seed": diffusion_seed,
                "selected_nodes": json.dumps(episode["selected_nodes"]),
                "final_spread": episode["final_spread"],
                "total_reward": episode["total_reward"],
                "selection_seconds": selection_seconds,
            }
        )
    return rows


def evaluate_degree(graph, seeds: list[int] = EVALUATION_SEEDS) -> list[dict[str, Any]]:
    """The fixed five highest-degree nodes, evaluated on every realisation."""
    env = SeedSelectionEnv(graph)
    fixed_selection = degree_selection(graph)
    rows = []
    for diffusion_seed in seeds:
        selection_start = time.perf_counter()
        actions = list(fixed_selection)
        selection_seconds = time.perf_counter() - selection_start
        episode = _finish_episode(env, actions, diffusion_seed)
        rows.append(
            {
                "method": "degree",
                "train_seed": -1,
                "diffusion_seed": diffusion_seed,
                "selected_nodes": json.dumps(episode["selected_nodes"]),
                "final_spread": episode["final_spread"],
                "total_reward": episode["total_reward"],
                "selection_seconds": selection_seconds,
            }
        )
    return rows


@torch.no_grad()
def evaluate_neural_policy(
    policy: torch.nn.Module,
    method: str,
    train_seed: int,
    graph,
    seeds: list[int] = EVALUATION_SEEDS,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Greedy sequential selection for one checkpoint on every realisation."""
    policy.eval()
    env = SeedSelectionEnv(graph)
    rows = []
    for diffusion_seed in seeds:
        observation, mask = env.reset(diffusion_seed)
        actions = []
        selection_seconds = 0.0
        for _ in range(5):
            start = time.perf_counter()
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
            logits = policy(observation_tensor)
            action = greedy_action(logits, mask_tensor)
            selection_seconds += time.perf_counter() - start

            observation, reward, terminated, info = env.step(action)
            mask = env.action_mask
            actions.append(action)
        if not terminated:
            raise RuntimeError("neural evaluation did not terminate after five selections")
        rows.append(
            {
                "method": method,
                "train_seed": train_seed,
                "diffusion_seed": diffusion_seed,
                "selected_nodes": json.dumps(actions),
                "final_spread": int(info["final_spread"]),
                "total_reward": float(reward),
                "selection_seconds": selection_seconds,
            }
        )
    return rows


def build_evaluation_records(
    graph,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], float]]:
    """Run all 800 evaluation episodes and collect training times."""
    records: list[dict[str, Any]] = []
    records.extend(evaluate_random(graph))
    records.extend(evaluate_degree(graph))
    training_seconds: dict[tuple[str, int], float] = {}
    for method in LEARNED_METHODS:
        for train_seed in TRAINING_SEEDS:
            path = Path(checkpoint_dir) / f"{method}_seed{train_seed}.pt"
            policy, checkpoint = load_checkpoint(path, device=device)
            if checkpoint["method"] != method or checkpoint["train_seed"] != train_seed:
                raise ValueError(f"checkpoint metadata mismatch: {path}")
            training_seconds[(method, train_seed)] = float(checkpoint["training_seconds"])
            records.extend(
                evaluate_neural_policy(
                    policy, method, train_seed, graph, device=device
                )
            )
    return records, training_seconds


def build_summary(
    evaluation: pd.DataFrame, training_seconds: dict[tuple[str, int], float]
) -> pd.DataFrame:
    """Create the required eight-row summary from evaluation records."""
    rows = []
    for method in SUMMARY_METHODS:
        seeds = [-1] if method in {"random", "degree"} else list(TRAINING_SEEDS)
        for train_seed in seeds:
            subset = evaluation[
                (evaluation["method"] == method) & (evaluation["train_seed"] == train_seed)
            ]
            if len(subset) != 100:
                raise ValueError(f"expected 100 evaluation rows for {method}, seed {train_seed}")
            rows.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "spread_mean": float(subset["final_spread"].mean()),
                    "spread_std": float(subset["final_spread"].std(ddof=0)),
                    "reward_mean": float(subset["total_reward"].mean()),
                    "selection_seconds_mean": float(subset["selection_seconds"].mean()),
                    "training_seconds": (
                        0.0
                        if method in {"random", "degree"}
                        else float(training_seconds[(method, train_seed)])
                    ),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_architecture_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD across the three run-level mean spreads of each architecture."""
    rows = []
    for method in LEARNED_METHODS:
        run_means = summary.loc[summary["method"] == method, "spread_mean"].to_numpy()
        rows.append(
            {
                "method": method,
                "n_runs": len(run_means),
                "run_mean_spread_mean": float(np.mean(run_means)),
                "run_mean_spread_std": float(np.std(run_means, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _department_map(departments: pd.DataFrame) -> dict[int, str]:
    return dict(zip(departments["node_id"], departments["department"]))


def build_selection_analysis(
    evaluation: pd.DataFrame,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selected-user tables and department coverage.

    Neural and degree selections are deterministic given the fixed graph, so
    they repeat across diffusion seeds.  Random coverage is averaged over its
    100 episodes.
    """
    dept_of = _department_map(departments)
    original_of = dict(zip(nodes["node_id"], nodes["original_id"]))
    selected_rows = []
    coverage_rows = []

    fixed_methods = ("degree",) + LEARNED_METHODS
    for method in fixed_methods:
        method_rows = evaluation[evaluation["method"] == method]
        seeds = [-1] if method == "degree" else list(TRAINING_SEEDS)
        for train_seed in seeds:
            subset = method_rows[method_rows["train_seed"] == train_seed]
            selections = {
                tuple(json.loads(value)) for value in subset["selected_nodes"].tolist()
            }
            if len(selections) != 1:
                raise ValueError(
                    f"{method} seed {train_seed} has non-deterministic selections"
                )
            selection = list(next(iter(selections)))
            labels = [dept_of[node] for node in selection]
            coverage = len(set(labels))
            mean_spread = float(
                summary[
                    (summary["method"] == method) & (summary["train_seed"] == train_seed)
                ]["spread_mean"].iloc[0]
            )
            selected_rows.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "selection_order": json.dumps(selection),
                    "selected_original_ids": json.dumps(
                        [int(original_of[node]) for node in selection]
                    ),
                    "departments": json.dumps(labels),
                    "department_coverage": coverage,
                    "mean_final_spread": mean_spread,
                }
            )
            coverage_rows.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "episodes": len(subset),
                    "coverage_mean": float(coverage),
                    "coverage_std": 0.0,
                    "coverage_min": int(coverage),
                    "coverage_max": int(coverage),
                }
            )

    # Random: compute coverage separately for every one of the 100 episodes.
    random_rows = evaluation[evaluation["method"] == "random"]
    random_coverages = []
    for value in random_rows["selected_nodes"]:
        selection = json.loads(value)
        random_coverages.append(len({dept_of[node] for node in selection}))
    selected_rows.append(
        {
            "method": "random",
            "train_seed": -1,
            "selection_order": "varies by episode",
            "selected_original_ids": "varies by episode",
            "departments": "varies by episode",
            "department_coverage": float(np.mean(random_coverages)),
            "mean_final_spread": float(
                summary[summary["method"] == "random"]["spread_mean"].iloc[0]
            ),
        }
    )
    coverage_rows.append(
        {
            "method": "random",
            "train_seed": -1,
            "episodes": len(random_rows),
            "coverage_mean": float(np.mean(random_coverages)),
            "coverage_std": float(np.std(random_coverages, ddof=0)),
            "coverage_min": int(min(random_coverages)),
            "coverage_max": int(max(random_coverages)),
        }
    )

    return (
        pd.DataFrame(selected_rows),
        pd.DataFrame(coverage_rows),
    )


def save_evaluation_products(
    records: list[dict[str, Any]],
    graph,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    training_seconds: dict[tuple[str, int], float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write evaluation.csv, summary.csv, and supplementary analysis tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = pd.DataFrame(records, columns=EVALUATION_COLUMNS)
    summary = build_summary(evaluation, training_seconds)
    architecture = build_architecture_summary(summary)
    selected_users, coverage = build_selection_analysis(
        evaluation, nodes, departments, summary
    )

    evaluation.to_csv(output_dir / "evaluation.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    architecture.to_csv(output_dir / "architecture_summary.csv", index=False)
    selected_users.to_csv(output_dir / "selected_users.csv", index=False)
    coverage.to_csv(output_dir / "department_coverage.csv", index=False)
    return evaluation, summary, architecture, selected_users, coverage
