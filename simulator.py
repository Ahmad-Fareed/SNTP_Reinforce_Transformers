"""Reproducible Independent Cascade simulator.

Every undirected edge is represented by two directed arcs.  For a given
diffusion seed, one uniform draw is made for every arc in sorted ``(source,
target)`` order.  An arc is live when its draw is below the activation
probability.  The final spread is the number of distinct participants reachable
from the seeds through live arcs (the seeds themselves are always counted).

The sampled draws and live arcs are implementation details: only ``simulate``
is public, and the environment never exposes them to a policy.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


class IndependentCascadeSimulator:
    """Independent Cascade on the fixed undirected graph."""

    def __init__(self, graph: nx.Graph, diffusion_seed: int, probability: float = 0.10):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        self._n_nodes = graph.number_of_nodes()
        self._probability = float(probability)
        self._diffusion_seed = int(diffusion_seed)

        # Build the ordered directed arc list once for this graph.
        undirected = sorted((int(u), int(v)) for u, v in graph.edges())
        self._arcs = sorted(
            [(u, v) for u, v in undirected] + [(v, u) for u, v in undirected]
        )

        # Exactly one draw per arc, in sorted arc order.
        rng = np.random.default_rng(self._diffusion_seed)
        self._draws = rng.random(len(self._arcs))
        self._live_arcs = self._draws < self._probability

        # Adjacency of live arcs makes the reachability step straightforward.
        self._live_adjacency: list[list[int]] = [[] for _ in range(self._n_nodes)]
        for (source, target), is_live in zip(self._arcs, self._live_arcs):
            if is_live:
                self._live_adjacency[source].append(target)

    @property
    def n_arcs(self) -> int:
        """Number of directed arcs (public, contains no random information)."""
        return len(self._arcs)

    def simulate(self, seeds) -> int:
        """Return the number of distinct nodes reachable from ``seeds``.

        The seeds are included even when they have no outgoing live arc.  An
        empty seed set gives zero spread.
        """
        seed_list = [int(seed) for seed in seeds]
        if len(set(seed_list)) != len(seed_list):
            raise ValueError("seeds must be distinct")
        if any(seed < 0 or seed >= self._n_nodes for seed in seed_list):
            raise ValueError("seed IDs must lie in [0, N-1]")
        if not seed_list:
            return 0

        active = set(seed_list)
        queue = list(seed_list)
        while queue:
            node = queue.pop()
            for neighbour in self._live_adjacency[node]:
                if neighbour not in active:
                    active.add(neighbour)
                    queue.append(neighbour)
        return len(active)


def load_graph_edges(path: Path) -> list[tuple[int, int]]:
    """Read graph_edges.csv (used by tests and reporting)."""
    import pandas as pd

    edges = pd.read_csv(path)
    return [(int(s), int(t)) for s, t in edges[["source", "target"]].itertuples(index=False)]
