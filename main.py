"""End-to-end entry point for SE4095 Assignment 03.

Usage:
    python main.py --output results

The script validates the supplied dataset, computes the network analysis,
trains both policies with three seeds, evaluates all methods on the same 100
diffusion realisations, writes every required CSV/checkpoint/figure, runs the
acceptance checks, and builds the PDF report.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from checks import run_checks
from environment import SeedSelectionEnv
from evaluation import (
    EVALUATION_SEEDS,
    LEARNED_METHODS,
    SUMMARY_METHODS,
    build_evaluation_records,
    save_evaluation_products,
)
from models import architecture_config
from network import (
    build_graph,
    compute_network_analysis,
    load_dataset,
    save_network_products,
)
from plotting import generate_all_figures
from reinforce import (
    TRAINING_SEEDS,
    save_training_diagnostics,
    train_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_COLUMNS = ["method", "train_seed", "episode", "final_spread", "total_reward"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 03 end to end")
    parser.add_argument(
        "--output",
        default="results",
        help="output directory, relative to the project root (default: results)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device (default: cpu)",
    )
    return parser.parse_args(argv)


def _resolve_output(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_training(training: pd.DataFrame) -> None:
    if len(training) != 6000:
        raise RuntimeError(f"training.csv would contain {len(training)} rows, expected 6,000")
    if list(training.columns) != TRAINING_COLUMNS:
        raise RuntimeError("training.csv columns do not match the specification")
    for method in LEARNED_METHODS:
        for train_seed in TRAINING_SEEDS:
            subset = training[
                (training["method"] == method) & (training["train_seed"] == train_seed)
            ]
            if len(subset) != 1000:
                raise RuntimeError(f"{method} seed {train_seed} has {len(subset)} rows")
            if sorted(subset["episode"].tolist()) != list(range(1000)):
                raise RuntimeError(f"{method} seed {train_seed} episode numbering is wrong")
            if not np.allclose(
                subset["total_reward"].to_numpy(),
                subset["final_spread"].to_numpy() / 92,
            ):
                raise RuntimeError("training reward is not spread / 92")


def _validate_evaluation(evaluation: pd.DataFrame, summary: pd.DataFrame) -> None:
    if len(evaluation) != 800:
        raise RuntimeError(f"evaluation.csv would contain {len(evaluation)} rows, expected 800")
    if len(summary) != 8:
        raise RuntimeError(f"summary.csv would contain {len(summary)} rows, expected 8")
    expected = {
        ("random", -1),
        ("degree", -1),
        *{(method, seed) for method in LEARNED_METHODS for seed in TRAINING_SEEDS},
    }
    actual = set(
        (str(row.method), int(row.train_seed))
        for row in evaluation[["method", "train_seed"]].drop_duplicates().itertuples()
    )
    if actual != expected:
        raise RuntimeError(f"unexpected evaluation method/seed combinations: {actual}")


def _save_run_metadata(
    output_dir: Path, total_seconds: float, analysis: dict, training_seconds: dict
) -> None:
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "networkx": __import__("networkx").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "device": "cpu",
        "n_participants": analysis["n_participants"],
        "n_edges": analysis["n_edges"],
        "training_seconds": {
            f"{method}_seed{seed}": seconds
            for (method, seed), seconds in sorted(training_seconds.items())
        },
        "total_pipeline_seconds": total_seconds,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    output_dir = _resolve_output(args.output)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print("Loading and validating the supplied dataset...", flush=True)
    nodes, edges, departments = load_dataset(PROJECT_ROOT / "data")
    graph = build_graph(nodes, edges)
    analysis = compute_network_analysis(graph, nodes, departments)
    save_network_products(graph, nodes, departments, analysis, output_dir)
    print(
        f"  {analysis['n_participants']} participants, {analysis['n_edges']} edges, "
        f"density={analysis['density']:.6f}, components="
        f"{analysis['n_connected_components']}, isolates={analysis['n_isolates']}",
        flush=True,
    )
    print(f"  top degree: {analysis['top10_degree_centrality'][0]}", flush=True)

    training_results = []
    for method in LEARNED_METHODS:
        for train_seed in TRAINING_SEEDS:
            print(f"Training {method} with seed {train_seed}...", flush=True)
            result = train_policy(
                method,
                graph,
                train_seed=train_seed,
                output_dir=output_dir,
                device=args.device,
            )
            training_results.append(result)

    training_rows = [
        record for result in training_results for record in result["records"]
    ]
    training = pd.DataFrame(training_rows, columns=TRAINING_COLUMNS)
    _validate_training(training)
    training.to_csv(output_dir / "training.csv", index=False)
    save_training_diagnostics(training_results, output_dir / "training_diagnostics.json")

    print("Evaluating baselines and all six checkpoints...", flush=True)
    records, training_seconds = build_evaluation_records(
        graph, output_dir / "checkpoints", device=args.device
    )
    evaluation, summary, architecture, selected_users, coverage = save_evaluation_products(
        records,
        graph,
        nodes,
        departments,
        training_seconds,
        output_dir,
    )
    _validate_evaluation(evaluation, summary)

    print("Generating figures...", flush=True)
    generate_all_figures(
        training,
        summary,
        selected_users,
        graph,
        nodes,
        departments,
        figures_dir,
    )

    print("Running acceptance checks...", flush=True)
    diagnostics = [result["diagnostics"] for result in training_results]
    run_checks(
        graph=graph,
        training_diagnostics=diagnostics,
        checkpoint_dir=output_dir / "checkpoints",
        evaluation=evaluation,
        output_path=output_dir / "checks.json",
    )

    total_seconds = time.perf_counter() - start
    _save_run_metadata(
        output_dir,
        total_seconds,
        analysis,
        {(method, seed): seconds for (method, seed), seconds in training_seconds.items()},
    )

    # The report is generated last so that it uses the actual saved results.
    from report import generate_report

    print("Building report.pdf...", flush=True)
    report_path = generate_report(output_dir, PROJECT_ROOT / "report.pdf")
    total_seconds = time.perf_counter() - start
    _save_run_metadata(
        output_dir,
        total_seconds,
        analysis,
        training_seconds,
    )

    print("\nCompleted successfully.", flush=True)
    print(f"  training rows: {len(training)}", flush=True)
    print(f"  evaluation rows: {len(evaluation)}", flush=True)
    print(f"  summary rows: {len(summary)}", flush=True)
    print(f"  checkpoints: {len(list((output_dir / 'checkpoints').glob('*.pt')))}", flush=True)
    print(f"  report: {report_path}", flush=True)
    print(f"  total wall time: {total_seconds:.1f}s", flush=True)
    print("\nSummary:\n" + summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
