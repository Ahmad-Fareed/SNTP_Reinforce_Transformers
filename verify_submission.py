"""Final artifact verification for the Assignment 03 submission.

Run after ``python main.py --output results``:

    python verify_submission.py --results results --report report.pdf

The script is intentionally independent of the training code's internal state:
it re-reads the saved CSV files, checkpoints, figures, and PDF and checks the
assignment's required schemas, counts, seeds, metrics, and page structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from pypdf import PdfReader

EXPECTED_HASHES = {
    "nodes.csv": "22cbebc2094014e7b03257b11527c24804c2a63152ce3e9d7abc70f6fbe50076",
    "edges.csv": "b413b8416469105d42e53eec557ff5c7bc3d30179a0c52a0c5cd8c402cb50438",
    "departments.csv": "a2821b03da4dc96d887e5f39d2851130a87e41e94ad313bbef26f26b2927a7ee",
}
TRAINING_COLUMNS = ["method", "train_seed", "episode", "final_spread", "total_reward"]
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
METHODS = ("random", "degree", "reinforce_mlp", "reinforce_transformer")
SEEDS = (7, 21, 42)
EVALUATION_SEEDS = set(range(900000, 900100))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Verifier:
    def __init__(self, results: Path, report: Path):
        self.root = Path(__file__).resolve().parent
        self.results = results
        self.report = report
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, function) -> None:
        try:
            detail = function()
            self.checks.append((name, True, str(detail)))
        except Exception as exc:
            self.checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------
    def verify(self) -> int:
        self.check("supplied dataset hashes", self.data_hashes)
        self.check("graph_edges.csv", self.graph_edges)
        self.check("training.csv", self.training)
        self.check("evaluation.csv", self.evaluation)
        self.check("summary.csv", self.summary)
        self.check("six checkpoints", self.checkpoints)
        self.check("checks.json", self.checks_json)
        self.check("figures", self.figures)
        self.check("report.pdf", self.pdf)
        self.check("required source files", self.source_files)
        self.check("run metadata", self.metadata)

        width = max(len(name) for name, _, _ in self.checks)
        print("Final artifact verification")
        print("=" * 80)
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name:<{width}}  {detail}")
        failures = [name for name, passed, _ in self.checks if not passed]
        print("=" * 80)
        if failures:
            print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
            return 1
        print(f"PASSED: {len(self.checks)} artifact checks.")
        return 0

    # ------------------------------------------------------------------
    def data_hashes(self) -> str:
        for name, expected in EXPECTED_HASHES.items():
            path = self.root / "data" / name
            assert path.exists(), f"missing {path}"
            actual = _sha256(path)
            assert actual == expected, f"{name} hash mismatch: {actual}"
        return "nodes.csv, edges.csv, departments.csv match the supplied SHA-256 hashes"

    def graph_edges(self) -> str:
        edges = pd.read_csv(self.results / "graph_edges.csv")
        assert list(edges.columns) == ["source", "target"], list(edges.columns)
        assert len(edges) == 755, len(edges)
        expected = pd.read_csv(self.root / "data" / "edges.csv")
        assert set(map(tuple, edges.values)) == set(map(tuple, expected.values))
        assert (edges["source"] < edges["target"]).all()
        assert not edges.duplicated().any()
        assert edges[["source", "target"]].equals(
            edges[["source", "target"]].sort_values(["source", "target"]).reset_index(drop=True)
        )
        return "755 unique sorted rows, source < target, exactly the supplied edge set"

    def training(self) -> str:
        training = pd.read_csv(self.results / "training.csv")
        assert list(training.columns) == TRAINING_COLUMNS, list(training.columns)
        assert len(training) == 6000, len(training)
        assert set(training["method"]) == {"reinforce_mlp", "reinforce_transformer"}
        for method in ("reinforce_mlp", "reinforce_transformer"):
            for seed in SEEDS:
                subset = training[
                    (training["method"] == method) & (training["train_seed"] == seed)
                ]
                assert len(subset) == 1000, (method, seed, len(subset))
                assert sorted(subset["episode"].tolist()) == list(range(1000))
                assert subset["final_spread"].between(5, 92).all()
                assert np.allclose(
                    subset["total_reward"].to_numpy(),
                    subset["final_spread"].to_numpy() / 92,
                )
        return "6,000 rows; 6 runs x 1,000 episodes; reward = spread/92"

    def evaluation(self) -> str:
        evaluation = pd.read_csv(self.results / "evaluation.csv")
        assert list(evaluation.columns) == EVALUATION_COLUMNS, list(evaluation.columns)
        assert len(evaluation) == 800, len(evaluation)
        assert set(evaluation["method"]) == set(METHODS)
        expected_combinations = {("random", -1), ("degree", -1)}
        expected_combinations |= {(method, seed) for method in ("reinforce_mlp", "reinforce_transformer") for seed in SEEDS}
        actual_combinations = set(
            (str(row.method), int(row.train_seed))
            for row in evaluation[["method", "train_seed"]].drop_duplicates().itertuples()
        )
        assert actual_combinations == expected_combinations, actual_combinations
        for (method, train_seed), subset in evaluation.groupby(["method", "train_seed"]):
            assert len(subset) == 100, (method, train_seed, len(subset))
            assert set(subset["diffusion_seed"].tolist()) == EVALUATION_SEEDS
            for value in subset["selected_nodes"]:
                selected = json.loads(value)
                assert len(selected) == 5 and len(set(selected)) == 5, selected
                assert all(0 <= node < 92 for node in selected)
            assert np.allclose(
                subset["total_reward"].to_numpy(),
                subset["final_spread"].to_numpy() / 92,
            )
            assert (subset["selection_seconds"] >= 0).all()
        assert (evaluation[evaluation["method"].isin(["random", "degree"])]["train_seed"] == -1).all()
        return "800 rows; 8 groups x 100 common seeds; five distinct valid selections each"

    def summary(self) -> str:
        evaluation = pd.read_csv(self.results / "evaluation.csv")
        summary = pd.read_csv(self.results / "summary.csv")
        assert list(summary.columns) == SUMMARY_COLUMNS, list(summary.columns)
        assert len(summary) == 8, len(summary)
        expected = {("random", -1), ("degree", -1)}
        expected |= {(method, seed) for method in ("reinforce_mlp", "reinforce_transformer") for seed in SEEDS}
        actual = set(
            (str(row.method), int(row.train_seed))
            for row in summary[["method", "train_seed"]].itertuples()
        )
        assert actual == expected, actual
        for row in summary.itertuples(index=False):
            subset = evaluation[
                (evaluation["method"] == row.method)
                & (evaluation["train_seed"] == row.train_seed)
            ]
            assert len(subset) == 100
            assert np.isclose(row.spread_mean, subset["final_spread"].mean())
            assert np.isclose(row.spread_std, subset["final_spread"].std(ddof=0))
            assert np.isclose(row.reward_mean, subset["total_reward"].mean())
            assert np.isclose(row.selection_seconds_mean, subset["selection_seconds"].mean())
            if row.method in {"random", "degree"}:
                assert row.train_seed == -1
                assert row.training_seconds == 0.0
            else:
                assert row.training_seconds > 0
        return "8 rows; all metrics reproduce evaluation.csv with population SD"

    def checkpoints(self) -> str:
        directory = self.results / "checkpoints"
        expected = {
            f"{method}_seed{seed}.pt"
            for method in ("reinforce_mlp", "reinforce_transformer")
            for seed in SEEDS
        }
        actual = {path.name for path in directory.glob("*.pt")}
        assert actual == expected, actual
        for filename in sorted(expected):
            checkpoint = torch.load(directory / filename, map_location="cpu", weights_only=False)
            method = filename.rsplit("_seed", 1)[0]
            seed = int(filename.rsplit("_seed", 1)[1].split(".")[0])
            assert checkpoint["method"] == method
            assert checkpoint["train_seed"] == seed
            assert len(checkpoint["state_dict"]) > 0
            assert checkpoint["architecture"]["type"] in {"mlp", "transformer"}
        return f"{len(expected)} checkpoints load with matching method/seed/architecture metadata"

    def checks_json(self) -> str:
        with (self.results / "checks.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["all_passed"] is True
        required = {
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
        }
        assert required.issubset(payload["checks"])
        assert all(payload["checks"][name]["passed"] for name in required)
        return f"all_passed=true; {len(payload['checks'])} checks passed, all ten mandatory checks present"

    def figures(self) -> str:
        expected = {
            "network_analysis.png",
            "learning_curves.png",
            "final_spread_comparison.png",
            "transformer_selection.png",
        }
        directory = self.results / "figures"
        actual = {path.name for path in directory.glob("*.png")}
        assert expected.issubset(actual), actual
        for name in expected:
            with PILImage.open(directory / name) as image:
                assert image.width >= 800 and image.height >= 500, (name, image.size)
        return "four PNG figures present at readable resolution"

    def pdf(self) -> str:
        assert self.report.exists(), self.report
        reader = PdfReader(str(self.report))
        assert len(reader.pages) >= 5, len(reader.pages)
        texts = [page.extract_text() or "" for page in reader.pages]
        full = "\n".join(texts)
        for phrase in (
            "Sequential Influence Maximization",
            "Independent Cascade",
            "REINFORCE",
            "Controlled evaluation",
            "References",
            "Génois",
            "Williams",
        ):
            assert phrase in full, f"missing searchable phrase: {phrase}"
        references_page = next(
            (index for index, text in enumerate(texts) if "References" in text),
            None,
        )
        assert references_page is not None
        body_pages = references_page
        assert 4 <= body_pages <= 6, f"body pages excluding references = {body_pages}"
        assert len(full) > 15000, len(full)
        return (
            f"searchable PDF with {len(reader.pages)} pages; "
            f"{body_pages} body pages excluding references"
        )

    def source_files(self) -> str:
        expected = [
            "main.py",
            "environment.py",
            "models.py",
            "reinforce.py",
            "requirements.txt",
            "README.md",
            "report.pdf",
            "data/nodes.csv",
            "data/edges.csv",
            "data/departments.csv",
            "data/dataset_README.md",
        ]
        for name in expected:
            path = self.root / name
            assert path.exists() and path.stat().st_size > 0, name
        return "all required source, dataset, and root report files exist"

    def metadata(self) -> str:
        with (self.results / "run_metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["n_participants"] == 92
        assert metadata["n_edges"] == 755
        assert len(metadata["training_seconds"]) == 6
        assert metadata["total_pipeline_seconds"] > 0
        return (
            f"92 participants, 755 edges, six training times, "
            f"total pipeline {metadata['total_pipeline_seconds']:.1f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the final Assignment 03 artifacts")
    parser.add_argument("--results", default="results")
    parser.add_argument("--report", default="report.pdf")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    results = Path(args.results)
    report = Path(args.report)
    if not results.is_absolute():
        results = root / results
    if not report.is_absolute():
        report = root / report
    return Verifier(results, report).verify()


if __name__ == "__main__":
    raise SystemExit(main())
