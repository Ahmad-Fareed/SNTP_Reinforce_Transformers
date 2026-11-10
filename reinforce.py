"""REINFORCE training for the two required policy architectures.

The implementation follows the assignment directly:

* 20 complete episodes are collected per update;
* the loss is ``-(1/20) * sum_i (R_i - b) * sum_t log pi(a_it | s_it)``;
* the baseline is fixed during collection and the update, then moved towards
  the batch mean reward;
* exactly 50 Adam updates are performed for 1,000 episodes.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from environment import SeedSelectionEnv
from models import (
    architecture_config,
    build_policy,
    masked_probabilities,
)

TRAINING_SEEDS = (7, 21, 42)
EPISODES_PER_RUN = 1000
EPISODES_PER_UPDATE = 20
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 1.0
MAX_GRAD_NORM = 1.0
UPDATE_COUNT = EPISODES_PER_RUN // EPISODES_PER_UPDATE


def set_training_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch at the start of every training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device | str) -> None:
    """Synchronise CUDA work before timing (no-op on CPU)."""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _tensors(observation: np.ndarray, mask: np.ndarray, device: torch.device):
    return (
        torch.as_tensor(observation, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


def _checkpoint_path(output_dir: Path, method: str, train_seed: int) -> Path:
    return Path(output_dir) / "checkpoints" / f"{method}_seed{train_seed}.pt"


def train_policy(
    method: str,
    graph,
    train_seed: int,
    output_dir: Path,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one policy for 1,000 episodes and save its final checkpoint."""
    if method not in {"reinforce_mlp", "reinforce_transformer"}:
        raise ValueError(f"unknown method: {method}")
    if train_seed not in TRAINING_SEEDS:
        raise ValueError(f"training seed must be one of {TRAINING_SEEDS}")

    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    set_training_seeds(train_seed)

    policy = build_policy(method).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    env = SeedSelectionEnv(graph, activation_probability=0.10)
    baseline = 0.0

    records: list[dict[str, Any]] = []
    losses: list[float] = []
    gradient_norms: list[float] = []
    gradient_norms_after_clip: list[float] = []
    batch_mean_rewards: list[float] = []
    baselines_before_update: list[float] = []
    batch_episode_counts: list[int] = []
    diffusion_seeds: list[int] = []
    all_finite_gradients = True

    synchronize(device)
    training_start = time.perf_counter()

    for update in range(UPDATE_COUNT):
        episode_log_probs: list[torch.Tensor] = []
        batch_rewards: list[float] = []

        # ---- Collect 20 complete episodes (no optimizer step here). ----
        for offset in range(EPISODES_PER_UPDATE):
            episode = update * EPISODES_PER_UPDATE + offset
            diffusion_seed = 1_000_000 + 10_000 * train_seed + episode
            diffusion_seeds.append(diffusion_seed)
            observation, mask = env.reset(diffusion_seed)
            selected_log_probs: list[torch.Tensor] = []

            for _ in range(env.budget):
                observation_tensor, mask_tensor = _tensors(observation, mask, torch.device(device))
                logits = policy(observation_tensor)
                probabilities = masked_probabilities(logits, mask_tensor)
                distribution = torch.distributions.Categorical(probs=probabilities)
                action = distribution.sample()
                selected_log_probs.append(distribution.log_prob(action))

                observation, reward, terminated, info = env.step(int(action.item()))
                mask = env.action_mask

            if not terminated:
                raise RuntimeError("episode did not terminate after five selections")
            final_spread = int(info["final_spread"])
            total_reward = float(reward)
            episode_log_probs.append(torch.stack(selected_log_probs).sum())
            batch_rewards.append(total_reward)
            records.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "episode": episode,
                    "final_spread": final_spread,
                    "total_reward": total_reward,
                }
            )

        # ---- One update from the complete batch; baseline stays fixed. ----
        baseline_before = float(baseline)
        weighted_terms = [
            -(reward - baseline_before) * log_prob
            for reward, log_prob in zip(batch_rewards, episode_log_probs)
        ]
        loss = torch.stack(weighted_terms).mean()

        optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        finite_gradients = all(bool(torch.isfinite(g).all()) for g in gradients)
        all_finite_gradients = all_finite_gradients and finite_gradients
        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        after_clip = torch.sqrt(
            sum((p.grad.detach() ** 2).sum() for p in policy.parameters() if p.grad is not None)
        )
        optimizer.step()

        # The baseline is updated only after the optimizer step, using the
        # mean reward of the batch that has just been discarded.
        batch_mean_reward = float(np.mean(batch_rewards))
        baseline = 0.9 * baseline_before + 0.1 * batch_mean_reward

        loss_value = float(loss.detach())
        losses.append(loss_value)
        gradient_norms.append(float(total_norm.detach()))
        gradient_norms_after_clip.append(float(after_clip.detach()))
        batch_mean_rewards.append(batch_mean_reward)
        baselines_before_update.append(baseline_before)
        batch_episode_counts.append(len(batch_rewards))

        if not np.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at update {update} for {method} seed {train_seed}")
        if not finite_gradients:
            raise RuntimeError(
                f"non-finite gradient at update {update} for {method} seed {train_seed}"
            )

        if verbose and (update + 1) % 10 == 0:
            print(
                f"  {method} seed {train_seed}: update {update + 1}/{UPDATE_COUNT}, "
                f"loss={loss_value:.4f}, batch mean reward={batch_mean_reward:.4f}, "
                f"baseline={baseline:.4f}",
                flush=True,
            )

    synchronize(device)
    training_seconds = time.perf_counter() - training_start

    if len(records) != EPISODES_PER_RUN:
        raise RuntimeError("training run did not produce exactly 1,000 episodes")
    if len(losses) != UPDATE_COUNT:
        raise RuntimeError("training run did not perform exactly 50 optimizer updates")

    # Verify the recorded baseline sequence matches the prescribed update.
    expected_baseline = 0.0
    baseline_sequence_ok = True
    for old_baseline, batch_mean in zip(baselines_before_update, batch_mean_rewards):
        if not np.isclose(old_baseline, expected_baseline, atol=1e-12):
            baseline_sequence_ok = False
        expected_baseline = 0.9 * expected_baseline + 0.1 * batch_mean
    baseline_sequence_ok = baseline_sequence_ok and np.isclose(
        baseline, expected_baseline, atol=1e-12
    )

    diagnostics = {
        "method": method,
        "train_seed": train_seed,
        "episodes": len(records),
        "episodes_per_update": EPISODES_PER_UPDATE,
        "batch_episode_counts": batch_episode_counts,
        "optimizer_steps": len(losses),
        "learning_rate": LEARNING_RATE,
        "discount_factor": DISCOUNT_FACTOR,
        "max_grad_norm": MAX_GRAD_NORM,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "gradient_norms_after_clip": gradient_norms_after_clip,
        "batch_mean_rewards": batch_mean_rewards,
        "baselines_before_update": baselines_before_update,
        "diffusion_seeds": diffusion_seeds,
        "final_baseline": float(baseline),
        "baseline_sequence_ok": bool(baseline_sequence_ok),
        "finite_losses": bool(all(np.isfinite(losses))),
        "finite_gradients": bool(all_finite_gradients and all(np.isfinite(gradient_norms))),
        "max_gradient_norm_after_clip": float(max(gradient_norms_after_clip)),
        "gradients_clipped_to_max": bool(
            max(gradient_norms_after_clip) <= MAX_GRAD_NORM + 1e-6
        ),
        "training_seconds": float(training_seconds),
    }

    checkpoint = {
        "method": method,
        "train_seed": train_seed,
        "architecture": architecture_config(method),
        "state_dict": policy.state_dict(),
        "training_settings": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "discount_factor": DISCOUNT_FACTOR,
            "episodes_per_update": EPISODES_PER_UPDATE,
            "optimizer_steps": UPDATE_COUNT,
            "episodes": EPISODES_PER_RUN,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "final_baseline": float(baseline),
        "training_seconds": float(training_seconds),
    }
    torch.save(checkpoint, _checkpoint_path(output_dir, method, train_seed))

    if not diagnostics["gradients_clipped_to_max"]:
        raise RuntimeError("gradient clipping did not bound the gradient norm")

    return {
        "method": method,
        "train_seed": train_seed,
        "records": records,
        "training_seconds": float(training_seconds),
        "baseline": float(baseline),
        "diagnostics": diagnostics,
        "policy": policy,
    }


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a policy from a saved checkpoint and load its parameters."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    method = checkpoint["method"]
    policy = build_policy(method).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, checkpoint


def save_training_diagnostics(results: list[dict[str, Any]], path: Path) -> None:
    """Write compact per-run diagnostic records (no full trajectories)."""
    compact = []
    for result in results:
        diagnostics = dict(result["diagnostics"])
        compact.append(diagnostics)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(compact, handle, indent=2)
