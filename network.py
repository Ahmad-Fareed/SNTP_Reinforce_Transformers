"""Dataset loading, graph construction, and descriptive network analysis.

The supplied SocioPatterns package is treated as read-only.  This module checks
the published SHA-256 hashes before using the files, builds the fixed
undirected/unweighted graph, and saves the analysis products used by the report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

N_NODES = 92
EXPECTED_FILES = {
    "nodes.csv": "22cbebc2094014e7b03257b11527c24804c2a63152ce3e9d7abc70f6fbe50076",
    "edges.csv": "b413b8416469105d42e53eec557ff5c7bc3d30179a0c52a0c5cd8c402cb50438",
    "departments.csv": "a2821b03da4dc96d887e5f39d2851130a87e41e94ad313bbef26f26b2927a7ee",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_supplied_files(data_dir: Path) -> None:
    """Raise if the instructor's CSV files were modified or are missing."""
    for name, expected in EXPECTED_FILES.items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing supplied dataset file: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{name} does not match the supplied package "
                f"(expected {expected}, got {actual})."
            )


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate nodes, edges, and department labels."""
    data_dir = Path(data_dir)
    verify_supplied_files(data_dir)

    nodes = pd.read_csv(data_dir / "nodes.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    departments = pd.read_csv(data_dir / "departments.csv")

    if list(nodes.columns) != ["node_id", "original_id"]:
        raise ValueError("nodes.csv must have columns node_id,original_id")
    if list(edges.columns) != ["source", "target"]:
        raise ValueError("edges.csv must have columns source,target")
    if list(departments.columns) != ["node_id", "department"]:
        raise ValueError("departments.csv must have columns node_id,department")

    expected_ids = list(range(N_NODES))
    if len(nodes) != N_NODES or sorted(nodes["node_id"].tolist()) != expected_ids:
        raise ValueError("nodes.csv must contain exactly the node IDs 0-91")
    if nodes["node_id"].duplicated().any():
        raise ValueError("nodes.csv contains duplicate node IDs")
    if not nodes["original_id"].is_monotonic_increasing:
        raise ValueError("original numeric IDs must be sorted by the mapping")
    if sorted(departments["node_id"].tolist()) != expected_ids:
        raise ValueError("departments.csv must contain every participant exactly once")

    if edges[["source", "target"]].isna().any().any():
        raise ValueError("edges.csv contains missing values")
    if not pd.api.types.is_integer_dtype(edges["source"]) or not pd.api.types.is_integer_dtype(
        edges["target"]
    ):
        raise ValueError("edge endpoints must be integers")
    if (edges["source"] == edges["target"]).any():
        raise ValueError("self-loops are not allowed")
    if (edges["source"] >= edges["target"]).any():
        raise ValueError("every edge must satisfy source < target")
    if edges.duplicated(subset=["source", "target"]).any():
        raise ValueError("duplicate edges are not allowed")
    if not edges[["source", "target"]].apply(
        lambda row: (row["source"], row["target"])
        == tuple(sorted((row["source"], row["target"]))),
        axis=1,
    ).all():
        raise ValueError("edges must be sorted by source then target")
    endpoint_ids = set(edges["source"]) | set(edges["target"])
    if not endpoint_ids.issubset(set(expected_ids)):
        raise ValueError("edges.csv contains node IDs outside 0-91")

    return nodes, edges, departments


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build the fixed undirected, unweighted graph.

    All 92 nodes are added before edges so that isolates would be retained.
    """
    graph = nx.Graph()
    graph.add_nodes_from(sorted(nodes["node_id"].tolist()))
    graph.add_edges_from(
        sorted((int(s), int(t)) for s, t in edges[["source", "target"]].itertuples(index=False))
    )
    return graph


def compute_network_analysis(
    graph: nx.Graph, nodes: pd.DataFrame, departments: pd.DataFrame
) -> dict:
    """Compute the statistics required by the assignment."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = dict(graph.degree())
    degree_centrality = {node: degree / (n - 1) for node, degree in degrees.items()}
    clustering = nx.clustering(graph)
    components = [sorted(component) for component in nx.connected_components(graph)]
    density = 2.0 * m / (n * (n - 1))

    original_id = dict(zip(nodes["node_id"], nodes["original_id"]))
    department_of = dict(zip(departments["node_id"], departments["department"]))
    top10 = sorted(
        graph.nodes(),
        key=lambda node: (-degrees[node], node),
    )[:10]
    top10_rows = [
        {
            "node_id": int(node),
            "original_id": int(original_id[node]),
            "department": department_of[node],
            "degree": int(degrees[node]),
            "degree_centrality": float(degree_centrality[node]),
            "clustering_coefficient": float(clustering[node]),
        }
        for node in top10
    ]

    degree_rows = [
        {
            "node_id": int(node),
            "original_id": int(original_id[node]),
            "department": department_of[node],
            "degree": int(degrees[node]),
            "degree_centrality": float(degree_centrality[node]),
            "clustering_coefficient": float(clustering[node]),
        }
        for node in range(n)
    ]

    return {
        "n_participants": n,
        "n_edges": m,
        "density": float(density),
        "n_connected_components": len(components),
        "component_sizes": [len(component) for component in components],
        "n_isolates": int(sum(1 for _, degree in graph.degree() if degree == 0)),
        "mean_degree": float(np.mean(list(degrees.values()))),
        "max_degree": int(max(degrees.values())),
        "min_degree": int(min(degrees.values())),
        "average_clustering": float(nx.average_clustering(graph)),
        "degree_centrality_sum": float(sum(degree_centrality.values())),
        "top10_degree_centrality": top10_rows,
        "degree_table": degree_rows,
        "department_counts": {
            str(k): int(v)
            for k, v in departments["department"].value_counts().sort_index().items()
        },
    }


def save_network_products(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    analysis: dict,
    output_dir: Path,
) -> None:
    """Write graph_edges.csv, analysis JSON, and degree tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    edge_rows = sorted((int(u), int(v)) for u, v in graph.edges())
    pd.DataFrame(edge_rows, columns=["source", "target"]).to_csv(
        output_dir / "graph_edges.csv", index=False
    )
    pd.DataFrame(analysis["degree_table"]).to_csv(
        output_dir / "degree_centrality.csv", index=False
    )
    pd.DataFrame(analysis["top10_degree_centrality"]).to_csv(
        output_dir / "top10_degree.csv", index=False
    )
    serializable = {
        key: value for key, value in analysis.items() if key not in {"degree_table"}
    }
    with (output_dir / "network_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    plot_network_analysis(graph, nodes, departments, analysis, figures_dir)


def _department_colors(departments: pd.DataFrame) -> dict[str, str]:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    names = sorted(departments["department"].unique())
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def plot_network_analysis(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    analysis: dict,
    figures_dir: Path,
) -> Path:
    """Draw one readable department-coloured network with major hubs labelled."""
    dept_of = dict(zip(departments["node_id"], departments["department"]))
    colors = _department_colors(departments)
    node_colors = [colors[dept_of[node]] for node in graph.nodes()]
    degrees = dict(graph.degree())
    node_sizes = [40 + 10 * degrees[node] for node in graph.nodes()]

    # A fixed seed keeps the figure reproducible.
    positions = nx.spring_layout(graph, seed=42, k=0.42, iterations=300, weight=None)
    top10 = [int(row["node_id"]) for row in analysis["top10_degree_centrality"]]
    label_nodes = top10[:5]

    fig, ax = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#9aa5b1", width=0.7, alpha=0.20)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        nodelist=top10,
        node_color=[node_colors[node] for node in top10],
        node_size=[node_sizes[node] for node in top10],
        edgecolors="#111111",
        linewidths=1.1,
        alpha=0.98,
    )
    center = np.mean(np.array(list(positions.values())), axis=0)
    label_positions = {}
    for node in label_nodes:
        node_position = np.array(positions[node])
        direction = node_position - center
        distance = np.linalg.norm(direction)
        offset = 0.16 * direction / distance if distance else np.array([0.0, 0.16])
        label_positions[node] = tuple(node_position + offset)
    if 12 in label_positions:
        x, y = label_positions[12]
        label_positions[12] = (x, y + 0.045)
    if 27 in label_positions:
        x, y = label_positions[27]
        label_positions[27] = (x, y - 0.045)
    for node in label_nodes:
        ax.annotate(
            str(node),
            xy=positions[node],
            xytext=label_positions[node],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#555555", linewidth=0.8, shrinkA=1, shrinkB=4),
            zorder=6,
        )
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10,
            label=name,
        )
        for name, color in colors.items()
    ]
    ax.legend(
        handles=handles, title="Department", loc="lower left", frameon=True, fontsize=10.5,
        title_fontsize=10.5,
    )
    ax.set_title(
        "SocioPatterns workplace contact network\n"
        f"({analysis['n_participants']} participants, {analysis['n_edges']} edges, "
        f"density={analysis['density']:.4f})",
        fontsize=13.5,
    )
    ax.set_axis_off()
    ax.text(
        0.99,
        0.01,
        "Node size = degree; black outline = top 10; labels = five largest hubs (node IDs).",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="#444444",
    )
    path = Path(figures_dir) / "network_analysis.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    return path
