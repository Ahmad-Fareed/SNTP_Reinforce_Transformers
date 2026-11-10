"""MLP and Transformer policies for sequential seed selection.

Both policies map the same ``(N, 5)`` observation matrix to one logit per
participant.  Action masking happens *after* the policy has processed all
participants.  In particular, the Transformer always attends over all 92
tokens, including already-selected participants; selected tokens are hidden
only when the final action distribution is formed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def masked_probabilities(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over valid actions only.

    ``action_mask`` is True for selectable participants.  Masked logits are set
    to negative infinity, so they receive exactly zero probability.
    """
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask must have the same shape")
    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    if not bool(mask.any()):
        raise ValueError("at least one action must be available")
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(masked_logits, dim=-1)


def masked_log_probabilities(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid actions, used by REINFORCE."""
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask must have the same shape")
    mask = action_mask.to(dtype=torch.bool, device=logits.device)
    if not bool(mask.any()):
        raise ValueError("at least one action must be available")
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def greedy_action(logits: torch.Tensor, action_mask: torch.Tensor) -> int:
    """Highest-probability valid action; exact ties go to the smaller ID."""
    probabilities = masked_probabilities(logits, action_mask)
    # torch.argmax returns the first occurrence, i.e. the smallest participant ID.
    return int(torch.argmax(probabilities).item())


class MLPPolicy(nn.Module):
    """Shared per-participant MLP: 5 -> 64 -> 64 -> 1.

    The same weights are applied independently to every row of the observation
    matrix.  There is no communication between participants; sequential
    information enters only through the five observed features.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 5)`` observations to ``(N,)`` logits."""
        if observations.ndim != 2 or observations.shape[1] != 5:
            raise ValueError("MLP input must have shape (N, 5)")
        return self.network(observations).squeeze(-1)


class TransformerPolicy(nn.Module):
    """One token per participant, one standard post-norm encoder layer.

    Pipeline: Linear(5, 64) -> TransformerEncoderLayer -> Linear(64, 1).
    There are no positional embeddings, node-ID embeddings, causal masks, or
    extra final normalization.  Attention is unrestricted across all 92 tokens.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(5, 64)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        self.output_head = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map ``(N, 5)`` observations to ``(N,)`` logits.

        The observation matrix is treated as a batch of one sequence of N
        tokens.  No attention mask is supplied, so every token can attend to
        every other token, including selected participants.
        """
        if observations.ndim != 2 or observations.shape[1] != 5:
            raise ValueError("Transformer input must have shape (N, 5)")
        tokens = self.input_projection(observations).unsqueeze(0)  # (1, N, 64)
        encoded = self.encoder_layer(tokens)  # no src_mask: full self-attention
        return self.output_head(encoded).squeeze(0).squeeze(-1)  # (N,)


ARCHITECTURES = {
    "reinforce_mlp": MLPPolicy,
    "reinforce_transformer": TransformerPolicy,
}


def architecture_config(method: str) -> dict[str, Any]:
    """Settings stored in every checkpoint and used when rebuilding a policy."""
    if method == "reinforce_mlp":
        return {
            "type": "mlp",
            "input_features": 5,
            "hidden_sizes": [64, 64],
            "outputs": 1,
            "activation": "relu",
            "dropout": 0.0,
        }
    if method == "reinforce_transformer":
        return {
            "type": "transformer",
            "input_features": 5,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 128,
            "activation": "relu",
            "dropout": 0.0,
            "norm_first": False,
            "positional_embeddings": False,
            "node_id_embeddings": False,
            "causal_mask": False,
        }
    raise ValueError(f"unknown policy method: {method}")


def build_policy(method: str) -> nn.Module:
    """Create a fresh policy for one of the two required methods."""
    if method not in ARCHITECTURES:
        raise ValueError(f"unknown policy method: {method}")
    return ARCHITECTURES[method]()
