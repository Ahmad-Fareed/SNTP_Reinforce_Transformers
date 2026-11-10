"""Figures generated from the actual training and evaluation records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "random": "Random",
    "degree": "Degree",
    "reinforce_mlp": "REINFORCE MLP",
    "reinforce_transformer": "REINFORCE Transformer",
}
METHOD_COLORS = {
    "random": "#7f7f7f",
    "degree": "#2ca02c",
    "reinforce_mlp": "#1f77b4",
    "reinforce_transformer": "#d62728",
}
SEED_COLORS = {7: "#1f77b4", 21: "#ff7f0e", 42: "#2ca02c"}


def plot_learning_curves(training: pd.DataFrame, path: Path) -> Path:
    """Mean spread in non-overlapping 20-episode blocks for all six runs."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for axis, method in zip(axes, ["reinforce_mlp", "reinforce_transformer"]):
        subset = training[training["method"] == method]
        for train_seed in (7, 21, 42):
            run = subset[subset["train_seed"] == train_seed].sort_values("episode")
            blocks = run.assign(block=run["episode"] // 20).groupby("block")["final_spread"].mean()
            axis.plot(
                (blocks.index + 1) * 20,
                blocks.values,
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=SEED_COLORS[train_seed],
                label=f"seed {train_seed}",
            )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Training episode (end of 20-episode block)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean final spread within block")
    axes[0].legend(title="Training seed")
    figure.suptitle("Learning curves: mean final spread per 20-episode block", fontsize=13)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
    return Path(path)


def plot_final_spread_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """Mean final spread for all eight method/training-seed combinations."""
    order = [("random", -1), ("degree", -1)]
    order += [("reinforce_mlp", seed) for seed in (7, 21, 42)]
    order += [("reinforce_transformer", seed) for seed in (7, 21, 42)]
    labels, means, errors, colors = [], [], [], []
    for method, train_seed in order:
        row = summary[(summary["method"] == method) & (summary["train_seed"] == train_seed)]
        if row.empty:
            raise ValueError(f"missing summary row for {method}, seed {train_seed}")
        short_name = {
            "random": "Random",
            "degree": "Degree",
            "reinforce_mlp": "MLP",
            "reinforce_transformer": "Transformer",
        }[method]
        labels.append(
            short_name if train_seed == -1 else f"{short_name}\nseed {train_seed}"
        )
        means.append(float(row["spread_mean"].iloc[0]))
        errors.append(float(row["spread_std"].iloc[0]))
        colors.append(METHOD_COLORS[method])

    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    bars = axis.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
    degree_mean = float(summary[summary["method"] == "degree"]["spread_mean"].iloc[0])
    axis.axhline(degree_mean, color=METHOD_COLORS["degree"], linestyle="--", linewidth=1, alpha=0.7)
    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(x, labels, fontsize=8.5)
    axis.set_ylabel("Final spread (participants)")
    axis.set_title("Final spread comparison on diffusion seeds 900000-900099")
    axis.set_ylim(0, max(means) + max(errors) + 6)
    axis.grid(axis="y", alpha=0.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for method in ("random", "degree", "reinforce_mlp", "reinforce_transformer")
    ]
    axis.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    axis.text(
        0.99,
        0.98,
        "Error bars: population SD over 100 shared diffusion realisations",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
    return Path(path)


def plot_transformer_selection(
    graph: nx.Graph,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    selected_nodes: list[int],
    path: Path,
) -> Path:
    """Highlight the Transformer (seed 7) five selections on the network."""
    original_id = dict(zip(nodes["node_id"], nodes["original_id"]))
    department_of = dict(zip(departments["node_id"], departments["department"]))
    selected = set(int(node) for node in selected_nodes)
    if len(selected) != 5:
        raise ValueError("the Transformer selection must contain five distinct nodes")

    positions = nx.spring_layout(graph, seed=42, k=0.42, iterations=300, weight=None)
    figure, axis = plt.subplots(figsize=(8, 6.2), constrained_layout=True)
    nx.draw_networkx_edges(
        graph, positions, ax=axis, edge_color="#c9ced6", width=0.35, alpha=0.18
    )
    selected_edges = [
        (u, v) for u, v in graph.edges() if u in selected or v in selected
    ]
    nx.draw_networkx_edges(
        graph, positions, edgelist=selected_edges, ax=axis, edge_color="#d62728", width=1.1, alpha=0.55
    )
    non_selected = [node for node in graph.nodes() if node not in selected]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=non_selected,
        ax=axis,
        node_color="#b8c0cc",
        node_size=90,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.9,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sorted(selected),
        ax=axis,
        node_color="#d62728",
        node_size=520,
        edgecolors="black",
        linewidths=1.4,
    )
    labels = {
        node: f"{node} ({original_id[node]}, {department_of[node]})"
        for node in sorted(selected)
    }
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        ax=axis,
        font_size=12.5,
        font_color="#111111",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d62728", alpha=0.92),
    )
    axis.set_title(
        "Transformer policy (training seed 7): five selected participants\n"
        "Greedy selection; labels show node ID (original ID, department)",
        fontsize=14,
    )
    axis.set_axis_off()
    axis.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                       markeredgecolor="black", markersize=12, label="Selected by Transformer"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b8c0cc",
                       markeredgecolor="white", markersize=9, label="Other participants"),
        ],
        loc="lower left",
        fontsize=10.5,
    )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
    return Path(path)


def generate_all_figures(
    training: pd.DataFrame,
    summary: pd.DataFrame,
    selected_users: pd.DataFrame,
    graph: nx.Graph,
    nodes: pd.DataFrame,
    departments: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Path]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    transformer_row = selected_users[
        (selected_users["method"] == "reinforce_transformer")
        & (selected_users["train_seed"] == 7)
    ]
    if len(transformer_row) != 1:
        raise ValueError("missing Transformer seed-7 selection row")
    transformer_selection = json.loads(transformer_row["selection_order"].iloc[0])
    return {
        "learning_curves": plot_learning_curves(training, figures_dir / "learning_curves.png"),
        "final_spread_comparison": plot_final_spread_comparison(
            summary, figures_dir / "final_spread_comparison.png"
        ),
        "transformer_selection": plot_transformer_selection(
            graph,
            nodes,
            departments,
            transformer_selection,
            figures_dir / "transformer_selection.png",
        ),
    }
