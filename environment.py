"""Sequential seed-selection environment for the workplace network.

State contains one row per participant (ascending node ID) and exactly five
features.  Static graph features are computed once; selection features are
updated after each action.  Diffusion is executed only after the fifth and
final selection, so the first four rewards are always zero.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from simulator import IndependentCascadeSimulator

N_NODES = 92
BUDGET = 5


class SeedSelectionEnv:
    """Select five distinct participants; reward is final spread / N."""

    def __init__(self, graph: nx.Graph, activation_probability: float = 0.10):
        if graph.number_of_nodes() != N_NODES:
            raise ValueError(f"expected {N_NODES} participants, got {graph.number_of_nodes()}")
        self.graph = graph
        self.n_nodes = N_NODES
        self.budget = BUDGET
        self.activation_probability = float(activation_probability)

        # Static features, computed once on the fixed graph.
        self._degrees = np.array(
            [graph.degree(node) for node in range(self.n_nodes)], dtype=np.float64
        )
        self._degree_centrality = self._degrees / (self.n_nodes - 1)
        self._clustering = np.array(
            [nx.clustering(graph, node) for node in range(self.n_nodes)],
            dtype=np.float64,
        )
        self._neighbours = [
            sorted(int(neighbour) for neighbour in graph.neighbors(node))
            for node in range(self.n_nodes)
        ]

        self._observation = np.zeros((self.n_nodes, 5), dtype=np.float32)
        self._action_mask = np.ones(self.n_nodes, dtype=bool)
        self._selected: list[int] = []
        self._selected_set: set[int] = set()
        self._selected_neighbour_count = np.zeros(self.n_nodes, dtype=np.float64)
        self._diffusion_seed: int | None = None
        self._terminated = False
        self._final_spread: int | None = None

    # ------------------------------------------------------------------
    # Core Gym-like interface
    # ------------------------------------------------------------------
    def reset(self, diffusion_seed: int):
        """Start a new episode and return ``(observation, action_mask)``."""
        self._diffusion_seed = int(diffusion_seed)
        self._selected = []
        self._selected_set = set()
        self._selected_neighbour_count = np.zeros(self.n_nodes, dtype=np.float64)
        self._action_mask = np.ones(self.n_nodes, dtype=bool)
        self._terminated = False
        self._final_spread = None
        self._refresh_observation()
        return self._observation.copy(), self._action_mask.copy()

    def step(self, action: int):
        """Select one participant.

        Returns ``(observation, reward, terminated, info)``.  Invalid actions
        raise ``ValueError`` instead of being silently replaced.
        """
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset()")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer participant ID")
        action = int(action)
        if action < 0 or action >= self.n_nodes:
            raise ValueError(f"action must lie in [0, {self.n_nodes - 1}]")
        if action in self._selected_set:
            raise ValueError(f"participant {action} has already been selected")

        self._selected.append(action)
        self._selected_set.add(action)
        for neighbour in self._neighbours[action]:
            self._selected_neighbour_count[neighbour] += 1.0
        self._action_mask[action] = False
        self._refresh_observation()

        reward = 0.0
        info: dict[str, object] = {"selected_nodes": list(self._selected)}
        if len(self._selected) == self.budget:
            # Diffusion is triggered for the first time only after action five.
            simulator = IndependentCascadeSimulator(
                self.graph,
                diffusion_seed=self._diffusion_seed,
                probability=self.activation_probability,
            )
            self._final_spread = simulator.simulate(self._selected)
            reward = self._final_spread / self.n_nodes
            self._terminated = True
            info["final_spread"] = self._final_spread

        return self._observation.copy(), float(reward), self._terminated, info

    # ------------------------------------------------------------------
    # State accessors and feature construction
    # ------------------------------------------------------------------
    @property
    def action_mask(self) -> np.ndarray:
        """Boolean mask: True for unselected participants."""
        return self._action_mask.copy()

    @property
    def selected_nodes(self) -> list[int]:
        return list(self._selected)

    @property
    def remaining_budget(self) -> int:
        return self.budget - len(self._selected)

    @property
    def terminated(self) -> bool:
        return self._terminated

    def _refresh_observation(self) -> None:
        n = self.n_nodes
        self._observation[:, 0] = self._degree_centrality
        self._observation[:, 1] = self._clustering
        self._observation[:, 2] = 0.0
        self._observation[self._selected, 2] = 1.0

        # Selected-neighbour fraction: zero for isolates by construction.
        fractions = np.zeros(n, dtype=np.float64)
        non_isolated = self._degrees > 0
        fractions[non_isolated] = (
            self._selected_neighbour_count[non_isolated] / self._degrees[non_isolated]
        )
        self._observation[:, 3] = fractions
        self._observation[:, 4] = (self.budget - len(self._selected)) / self.budget
