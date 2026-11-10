"""Automated tests for controlled evaluation and summary construction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    EVALUATION_COLUMNS,
    EVALUATION_SEEDS,
    SUMMARY_COLUMNS,
    build_architecture_summary,
    build_selection_analysis,
    build_summary,
    degree_selection,
    evaluate_degree,
    evaluate_neural_policy,
    evaluate_random,
    save_evaluation_products,
)
from models import MLPPolicy  # noqa: E402
from network import build_graph, load_dataset  # noqa: E402


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.edges, cls.departments = load_dataset(ROOT / "data")
        cls.graph = build_graph(cls.nodes, cls.edges)

    def test_degree_selection_uses_degree_then_smaller_id(self):
        expected = sorted(self.graph.nodes(), key=lambda node: (-self.graph.degree(node), node))[:5]
        self.assertEqual(degree_selection(self.graph), expected)
        rows = evaluate_degree(self.graph)
        self.assertEqual(len(rows), 100)
        self.assertEqual({row["selected_nodes"] for row in rows}, {json.dumps(expected)})
        self.assertEqual({row["train_seed"] for row in rows}, {-1})

    def test_random_uses_prescribed_generator_and_distinct_users(self):
        rows = evaluate_random(self.graph)
        self.assertEqual(len(rows), 100)
        self.assertEqual([row["diffusion_seed"] for row in rows], EVALUATION_SEEDS)
        for row in rows[:5]:
            expected = np.random.default_rng(row["diffusion_seed"] + 10_000_000).choice(
                92, size=5, replace=False
            )
            self.assertEqual(json.loads(row["selected_nodes"]), [int(x) for x in expected])
        repeat = evaluate_random(self.graph)
        self.assertEqual(
            [row["selected_nodes"] for row in rows],
            [row["selected_nodes"] for row in repeat],
        )
        for row in rows:
            selected = json.loads(row["selected_nodes"])
            self.assertEqual(len(selected), 5)
            self.assertEqual(len(set(selected)), 5)

    def test_neural_evaluation_is_greedy_and_deterministic(self):
        policy = MLPPolicy()
        rows = evaluate_neural_policy(policy, "reinforce_mlp", 7, self.graph)
        self.assertEqual(len(rows), 100)
        self.assertEqual([row["diffusion_seed"] for row in rows], EVALUATION_SEEDS)
        self.assertEqual(len({row["selected_nodes"] for row in rows}), 1)
        for row in rows:
            selected = json.loads(row["selected_nodes"])
            self.assertEqual(len(set(selected)), 5)
            self.assertAlmostEqual(row["total_reward"], row["final_spread"] / 92, places=12)
            self.assertGreaterEqual(row["selection_seconds"], 0)

    def synthetic_evaluation(self):
        rows = []
        combinations = [("random", -1), ("degree", -1)]
        combinations += [("reinforce_mlp", seed) for seed in (7, 21, 42)]
        combinations += [("reinforce_transformer", seed) for seed in (7, 21, 42)]
        for index, (method, train_seed) in enumerate(combinations):
            for diffusion_seed in EVALUATION_SEEDS:
                spread = 50 + index + (diffusion_seed % 5)
                rows.append(
                    {
                        "method": method,
                        "train_seed": train_seed,
                        "diffusion_seed": diffusion_seed,
                        "selected_nodes": json.dumps([0, 1, 2, 3, 4]),
                        "final_spread": spread,
                        "total_reward": spread / 92,
                        "selection_seconds": 0.001 * (index + 1),
                    }
                )
        return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)

    def test_summary_has_exactly_eight_rows_and_population_sd(self):
        evaluation = self.synthetic_evaluation()
        training_seconds = {
            (method, seed): 10.0 + seed / 10
            for method in ("reinforce_mlp", "reinforce_transformer")
            for seed in (7, 21, 42)
        }
        summary = build_summary(evaluation, training_seconds)
        self.assertEqual(len(summary), 8)
        self.assertEqual(list(summary.columns), SUMMARY_COLUMNS)
        for method, train_seed in [("random", -1), ("degree", -1)]:
            row = summary[(summary["method"] == method) & (summary["train_seed"] == train_seed)].iloc[0]
            self.assertEqual(row["training_seconds"], 0.0)
            subset = evaluation[(evaluation["method"] == method) & (evaluation["train_seed"] == train_seed)]
            self.assertAlmostEqual(row["spread_mean"], subset["final_spread"].mean())
            self.assertAlmostEqual(row["spread_std"], subset["final_spread"].std(ddof=0))
        for method in ("reinforce_mlp", "reinforce_transformer"):
            for seed in (7, 21, 42):
                row = summary[(summary["method"] == method) & (summary["train_seed"] == seed)].iloc[0]
                self.assertAlmostEqual(row["training_seconds"], 10.0 + seed / 10)
        architecture = build_architecture_summary(summary)
        self.assertEqual(len(architecture), 2)
        for row in architecture.itertuples(index=False):
            run_means = summary[summary["method"] == row.method]["spread_mean"].to_numpy()
            self.assertAlmostEqual(row.run_mean_spread_mean, float(np.mean(run_means)))
            self.assertAlmostEqual(row.run_mean_spread_std, float(np.std(run_means, ddof=0)))

    def test_selection_analysis_reports_department_coverage(self):
        evaluation = self.synthetic_evaluation()
        training_seconds = {
            (method, seed): 1.0
            for method in ("reinforce_mlp", "reinforce_transformer")
            for seed in (7, 21, 42)
        }
        summary = build_summary(evaluation, training_seconds)
        selected, coverage = build_selection_analysis(
            evaluation, self.nodes, self.departments, summary
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(len(coverage), 8)
        self.assertIn("random", set(coverage["method"]))
        self.assertIn("degree", set(coverage["method"]))
        random_coverage = coverage[coverage["method"] == "random"].iloc[0]
        self.assertEqual(random_coverage["episodes"], 100)
        self.assertGreaterEqual(random_coverage["coverage_min"], 1)
        self.assertLessEqual(random_coverage["coverage_max"], 5)

    def test_saved_evaluation_csv_has_quoted_json_selection(self):
        evaluation = self.synthetic_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            training_seconds = {
                (method, seed): 1.0
                for method in ("reinforce_mlp", "reinforce_transformer")
                for seed in (7, 21, 42)
            }
            save_evaluation_products(
                evaluation.to_dict("records"),
                self.graph,
                self.nodes,
                self.departments,
                training_seconds,
                output,
            )
            saved = pd.read_csv(output / "evaluation.csv")
            self.assertEqual(len(saved), 800)
            self.assertEqual(list(saved.columns), EVALUATION_COLUMNS)
            self.assertEqual(json.loads(saved["selected_nodes"].iloc[0]), [0, 1, 2, 3, 4])
            self.assertEqual(len(pd.read_csv(output / "summary.csv")), 8)
            self.assertEqual(len(pd.read_csv(output / "architecture_summary.csv")), 2)
            self.assertEqual(len(pd.read_csv(output / "selected_users.csv")), 8)
            self.assertEqual(len(pd.read_csv(output / "department_coverage.csv")), 8)


if __name__ == "__main__":
    unittest.main()
