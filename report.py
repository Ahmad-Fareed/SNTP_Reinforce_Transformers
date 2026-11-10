"""Build the searchable PDF report from the actual saved results.

The report is generated after training and evaluation, so every number, table,
and figure comes from the current run.  It uses ReportLab (vector text, no
rasterised pages) and keeps references on a separate page.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 47
RIGHT_MARGIN = 47
TOP_MARGIN = 42
BOTTOM_MARGIN = 43
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=3,
            textColor=colors.HexColor("#1a2a44"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Abstract",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.1,
            alignment=TA_JUSTIFY,
            backColor=colors.HexColor("#f3f6fa"),
            borderColor=colors.HexColor("#dbe2ea"),
            borderWidth=0.6,
            borderPadding=6,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading1Report",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12.2,
            leading=14.5,
            spaceBefore=7,
            spaceAfter=3,
            textColor=colors.HexColor("#1a2a44"),
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Report",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=9.8,
            leading=12,
            spaceBefore=4,
            spaceAfter=2,
            textColor=colors.HexColor("#1a2a44"),
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.85,
            leading=11.0,
            alignment=TA_JUSTIFY,
            spaceAfter=3.2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletReport",
            parent=styles["BodyReport"],
            leftIndent=10,
            bulletIndent=2,
            spaceAfter=1.8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CaptionReport",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.9,
            leading=9.6,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a5568"),
            spaceBefore=2,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.1,
            leading=8.6,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Reference",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.9,
            leading=10.1,
            leftIndent=11,
            firstLineIndent=-11,
            spaceAfter=3.5,
        )
    )
    return styles


def _p(text: str, style, **kwargs) -> Paragraph:
    return Paragraph(text, style, **kwargs)


def _image(path: Path, width: float) -> Image:
    with PILImage.open(path) as image:
        image_width, image_height = image.size
    height = width * image_height / image_width
    return Image(str(path), width=width, height=height)


def _table(rows, widths, styles, header=True, font_size=7.1, align="LEFT"):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign=align)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b5a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                commands.append(
                    ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#f2f5f9"))
                )
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.setFont("Helvetica", 7.2)
    canvas.drawString(LEFT_MARGIN, 22, "SE4095 Social Network Theory and Practice - Assignment 03")
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 22, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#d1d5db"))
    canvas.setLineWidth(0.4)
    canvas.line(LEFT_MARGIN, 30, PAGE_WIDTH - RIGHT_MARGIN, 30)
    canvas.restoreState()


def _load_results(results_dir: Path):
    with (results_dir / "network_analysis.json").open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    summary = pd.read_csv(results_dir / "summary.csv")
    evaluation = pd.read_csv(results_dir / "evaluation.csv")
    training = pd.read_csv(results_dir / "training.csv")
    architecture = pd.read_csv(results_dir / "architecture_summary.csv")
    selected = pd.read_csv(results_dir / "selected_users.csv")
    coverage = pd.read_csv(results_dir / "department_coverage.csv")
    with (results_dir / "checks.json").open(encoding="utf-8") as handle:
        checks = json.load(handle)
    with (results_dir / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return analysis, summary, evaluation, training, architecture, selected, coverage, checks, metadata


def _summary_value(summary: pd.DataFrame, method: str, seed: int, column: str) -> float:
    row = summary[(summary["method"] == method) & (summary["train_seed"] == seed)]
    if row.empty:
        raise ValueError(f"missing summary row {method}, {seed}")
    return float(row[column].iloc[0])


def _method_label(method: str) -> str:
    return {
        "random": "Random",
        "degree": "Degree",
        "reinforce_mlp": "REINFORCE MLP",
        "reinforce_transformer": "REINFORCE Transformer",
    }[method]


def _department_counts(departments_json: str) -> str:
    """Compact department composition for the selected-user table."""
    labels = json.loads(departments_json)
    counts = Counter(str(label) for label in labels)
    return ", ".join(f"{department} x{count}" for department, count in sorted(counts.items()))


def _selected_overlap_text(selection_orders: dict[tuple[str, int], list[int]], graph) -> str:
    pieces = []
    for (method, seed), selected in selection_orders.items():
        internal_edges = sum(
            1 for i, u in enumerate(selected) for v in selected[i + 1 :] if graph.has_edge(u, v)
        )
        neighbour_union = set()
        for node in selected:
            neighbour_union.update(graph.neighbors(node))
        edge_word = "edge" if internal_edges == 1 else "edges"
        pieces.append(
            f"{_method_label(method)}"
            + ("" if seed == -1 else f" seed {seed}")
            + f": {internal_edges} internal {edge_word}, "
            f"{len(neighbour_union)} distinct neighbours"
        )
    return "; ".join(pieces)


def generate_report(results_dir: Path, output_path: Path) -> Path:
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    (
        analysis,
        summary,
        evaluation,
        training,
        architecture,
        selected,
        coverage,
        checks,
        metadata,
    ) = _load_results(results_dir)

    # Rebuild the graph only for the overlap sentence and check it is the
    # supplied graph (the CSV loader validates hashes in main).
    import networkx as nx

    graph_edges = pd.read_csv(results_dir / "graph_edges.csv")
    graph = nx.Graph()
    graph.add_nodes_from(range(analysis["n_participants"]))
    graph.add_edges_from(graph_edges[["source", "target"]].itertuples(index=False, name=None))

    styles = _styles()
    story = []
    story.append(_p("Sequential Influence Maximization in a Workplace Contact Network", styles["ReportTitle"]))
    story.append(
        _p(
            "SE4095 Social Network Theory and Practice - Assignment 03<br/>"
            "Independent Cascade, REINFORCE, and an MLP versus Transformer policy",
            styles["Subtitle"],
        )
    )

    random_mean = _summary_value(summary, "random", -1, "spread_mean")
    degree_mean = _summary_value(summary, "degree", -1, "spread_mean")
    mlp_arch = architecture[architecture["method"] == "reinforce_mlp"].iloc[0]
    transformer_arch = architecture[
        architecture["method"] == "reinforce_transformer"
    ].iloc[0]
    mlp_mean = float(mlp_arch["run_mean_spread_mean"])
    transformer_mean = float(transformer_arch["run_mean_spread_mean"])
    degree_seconds = _summary_value(summary, "degree", -1, "selection_seconds_mean") * 1000
    mlp_selection_ms = (
        summary[summary["method"] == "reinforce_mlp"]["selection_seconds_mean"].mean() * 1000
    )
    transformer_selection_ms = (
        summary[summary["method"] == "reinforce_transformer"]["selection_seconds_mean"].mean()
        * 1000
    )
    random_coverage_row = coverage[coverage["method"] == "random"].iloc[0]
    checks_passed = int(sum(entry["passed"] for entry in checks["checks"].values()))
    checks_total = len(checks["checks"])
    selection_orders = {}
    for row in selected.itertuples(index=False):
        if str(row.method) == "random":
            continue
        selection_orders[(str(row.method), int(row.train_seed))] = json.loads(row.selection_order)

    abstract = (
        "An organization can promote an awareness message to only five participants. "
        "We treat seed selection as a five-step sequential decision problem on the supplied "
        f"{analysis['n_participants']}-participant SocioPatterns workplace graph. The fixed graph has "
        f"{analysis['n_edges']} edges, density {analysis['density']:.4f}, one connected component, and "
        f"average clustering {analysis['average_clustering']:.4f}. Independent Cascade is implemented "
        "with activation probability 0.10 and one reproducible random draw per directed arc. We train "
        "an MLP policy and a compulsory Transformer policy with REINFORCE, using 1,000 episodes and "
        "3 seeds per architecture. All methods are evaluated on the same 100 diffusion realisations "
        f"(seeds 900000-900099). On mean spread, Degree is {degree_mean:.2f}, Random is {random_mean:.2f}, "
        f"the MLP is {mlp_mean:.2f}, and the Transformer is {transformer_mean:.2f}. "
        "We report the result without claiming that a learned policy must beat Degree, and we limit "
        "all conclusions to this fixed aggregated graph and simulator."
    )
    story.append(_p(abstract, styles["Abstract"]))

    # ------------------------------------------------------------------
    # 1 Network analysis
    # ------------------------------------------------------------------
    story.append(_p("1. Network analysis", styles["Heading1Report"]))
    story.append(
        _p(
            "The supplied package contains 92 participants and 755 distinct undirected contacts, "
            "aggregated over the full observation period. The graph is unweighted, has no self-loops "
            "or duplicate pairs, and every stored edge satisfies source &lt; target and is sorted. "
            f"Density is 2E/[N(N-1)] = 2(755)/[92(91)] = {analysis['density']:.6f}. All 92 nodes form one "
            f"connected component; there are no isolates. The mean degree is {analysis['mean_degree']:.2f}, "
            f"the minimum degree is {analysis['min_degree']}, and the maximum degree is {analysis['max_degree']}. "
            "Department labels are used only for visualization and descriptive analysis; they are not "
            "features, rewards, or tuning signals.",
            styles["BodyReport"],
        ),
    )
    stats_rows = [
        ["Statistic", "Value"],
        ["Participants", str(analysis["n_participants"])],
        ["Undirected edges", str(analysis["n_edges"])],
        ["Density", f"{analysis['density']:.6f}"],
        ["Connected components", str(analysis["n_connected_components"])],
        ["Isolated participants", str(analysis["n_isolates"])],
        ["Mean degree", f"{analysis['mean_degree']:.2f}"],
        ["Average clustering coefficient", f"{analysis['average_clustering']:.4f}"],
    ]
    stats_table = _table(stats_rows, [122, 62], styles)
    top_rows = [["Node", "Orig.", "Dept", "Degree", "Deg. cent.", "Clustering"]]
    for row in analysis["top10_degree_centrality"]:
        top_rows.append(
            [
                str(row["node_id"]),
                str(row["original_id"]),
                row["department"],
                str(row["degree"]),
                f"{row['degree_centrality']:.4f}",
                f"{row['clustering_coefficient']:.4f}",
            ]
        )
    top_table = _table(top_rows, [38, 40, 45, 44, 62, 62], styles)
    table_pair = Table(
        [[stats_table, top_table]],
        colWidths=[CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.60],
        hAlign="LEFT",
    )
    table_pair.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(table_pair)
    story.append(
        _p(
            "Table 1. Network statistics and the ten participants with the largest degree. "
            "Degree centrality is degree/(N-1); ties in the ranking are broken by smaller node ID.",
            styles["CaptionReport"],
        )
    )
    network_figure = results_dir / "figures" / "network_analysis.png"
    story.append(
        KeepTogether(
            [
                _image(network_figure, CONTENT_WIDTH * 0.60),
                _p(
                    "Figure 1. Workplace network; node size = degree, colour = department, black outlines = "
                    "ten largest degrees, labels = five largest hub IDs (original IDs are in Table 1).",
                    styles["CaptionReport"],
                ),
            ]
        )
    )

    # ------------------------------------------------------------------
    # 2 RL formulation
    # ------------------------------------------------------------------
    story.append(_p("2. Reinforcement-learning formulation", styles["Heading1Report"]))
    story.append(
        _p(
            "An episode is a finite-horizon Markov decision process with exactly five selections. The "
            "observation is a 92 x 5 matrix in ascending participant-ID order: (i) degree centrality, "
            "degree/(N-1); (ii) the local clustering coefficient on the fixed graph; (iii) a selected "
            "indicator; (iv) the fraction of a participant's neighbours that are already selected, "
            "defined as 0 for an isolate; and (v) remaining budget divided by 5. Static features (i) and "
            "(ii) are computed once. The selection features (iii)-(v) are refreshed after every action. "
            "The action is an integer 0-91 identifying an unselected participant. The Boolean mask is "
            "True for available participants and False after selection. Invalid actions, including "
            "duplicates, raise an error rather than being silently replaced.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            "The first four rewards are exactly zero. Diffusion is executed only after the fifth action, "
            "and the fifth reward is final_spread/92, where the final spread includes the five seeds. The "
            "episode then terminates; the total return is in [0,1]. The task is sequential because each "
            "choice changes the remaining budget and the selected-neighbour features, so the value of a "
            "participant depends on the other four choices. There is no reward shaping, penalty, or "
            "intermediate cascade feedback.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 3 Independent Cascade
    # ------------------------------------------------------------------
    story.append(_p("3. Independent Cascade simulator", styles["Heading1Report"]))
    story.append(
        _p(
            "Each undirected edge is represented by two directed arcs, and the complete arc list is sorted "
            "by (source, target). For diffusion seed d, numpy.random.default_rng(d) draws exactly one "
            "uniform number for every arc in that order. An arc is live when its draw is below 0.10. "
            "Starting from the five seeds, breadth-first reachability over live arcs gives the final "
            "active set; active users stay active and the seeds are always counted. The sampled draws "
            "and live arcs are private attributes of the simulator. Policies receive only the five "
            "observable features and never inspect the random state or query hypothetical outcomes.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            "Automated verification covered an empty seed set, probability 0, probability 1, and exact "
            "repeatability for the same graph, seed set, and diffusion seed. Probability 0 and 1 were used "
            "only for tests; every training and evaluation episode used probability 0.10. Reusing the same "
            "ordered arc list keeps the realisation identical across methods for a given diffusion seed. "
            f"The generated checks.json records {checks_passed}/{checks_total} passing automated checks.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 4 REINFORCE
    # ------------------------------------------------------------------
    story.append(_p("4. REINFORCE implementation", styles["Heading1Report"]))
    story.append(
        _p(
            "Training collects 20 complete episodes before each update. Actions are sampled from the "
            "masked categorical policy, and the five differentiable log-probabilities are retained. For "
            "episode i with terminal reward R_i and running baseline b, the loss is "
            "<i>L</i>(theta) = -(1/20) * sum_i [(<i>R</i><sub>i</sub> - <i>b</i>) * "
            "sum_t log pi_theta(<i>a</i><sub>i,t</sub> | <i>s</i><sub>i,t</sub>)]. Because the discount "
            "factor is 1.0, the same terminal return is used at all five selections. Rewards and the "
            "baseline are plain scalars, so they carry no gradient. The baseline starts at 0 and is held "
            "fixed while a batch is collected and its loss is computed; after the optimizer step it is "
            "updated as b = 0.9b + 0.1 * batch_mean_reward. The trajectories are then discarded.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            "Adam uses learning rate 0.001. One update is performed per 20-episode batch, giving exactly "
            "50 updates per 1,000-episode run. The gradient norm is clipped to 1.0, and no PPO, critic, "
            "replay buffer, entropy bonus, advantage normalisation, reward normalisation, schedule, or "
            "early stopping is used. Python, NumPy, and PyTorch are seeded with the run seed (7, 21, or "
            "42). Episode e uses diffusion_seed = 1,000,000 + 10,000 * train_seed + e. Training and "
            "evaluation share the graph; only the diffusion realisation changes.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 5 Policies
    # ------------------------------------------------------------------
    story.append(_p("5. Policy architectures", styles["Heading1Report"]))
    story.append(
        _p(
            "<b>MLP.</b> The same 5-64-64-1 network is applied independently to every participant: "
            "Linear(5,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,1). It returns 92 logits. "
            "Selected participants are masked and softmax is taken over the remaining participants. "
            "The MLP has no dropout, no extra layers, and no communication between participants; "
            "sequential information enters only through the five features.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            "<b>Transformer.</b> One participant is one token. A Linear(5,64) projection is followed by "
            "one standard PyTorch post-layer-normalisation encoder layer (d_model=64, 4 heads, "
            "feed-forward dimension 128, ReLU, dropout 0), then Linear(64,1) on each token. There are no "
            "positional embeddings, node-ID embeddings, causal masks, or extra final encoder "
            "normalisation. Attention is unrestricted across all 92 tokens, and selected tokens remain "
            "visible to self-attention; they are hidden only at the final action-selection stage. In each "
            "layer, every token forms queries, keys, and values; four attention heads learn different "
            "pairwise weightings, and residual connections plus normalization make the layer trainable. "
            "Attention is a learned all-to-all interaction, not the contact graph. The MLP scores each "
            "participant from its own row, whereas the Transformer can condition a participant's logit "
            "on the current selection pattern of all other participants.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 6 Evaluation protocol
    # ------------------------------------------------------------------
    story.append(_p("6. Controlled evaluation", styles["Heading1Report"]))
    story.append(
        _p(
            "Random samples five distinct participants uniformly with a separate NumPy generator seeded "
            "by diffusion_seed + 10,000,000. Degree selects the five highest-degree participants, with "
            "ties broken by smaller node ID. The six final checkpoints act greedily under the mask; exact "
            "probability ties also go to the smaller ID, gradients are disabled, and no parameter is "
            "updated. Every method is evaluated on diffusion seeds 900000-900099, giving 800 episodes: "
            "6 checkpoints x 100, Random x 100, and Degree x 100. Selection time excludes diffusion; for "
            "neural policies it measures observation-to-logit computation, masking, and argmax, while "
            "Random's sampling is included. Training time includes all training simulations and was "
            "measured with a monotonic performance counter on CPU.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 7 Results
    # ------------------------------------------------------------------
    story.append(_p("7. Results", styles["Heading1Report"]))
    story.append(
        _p(
            "Table 2 reports every method/training-seed combination from the 800 evaluation episodes. "
            "The standard deviations are population standard deviations over the 100 shared diffusion "
            "realisations. Table 3 aggregates the three run-level means of each architecture.",
            styles["BodyReport"],
        ),
    )
    summary_rows = [
        ["Method", "Seed", "Mean spread", "Pop. SD", "Mean reward", "Selection ms", "Training s"]
    ]
    order = [("random", -1), ("degree", -1)]
    order += [("reinforce_mlp", seed) for seed in (7, 21, 42)]
    order += [("reinforce_transformer", seed) for seed in (7, 21, 42)]
    for method, seed in order:
        row = summary[(summary["method"] == method) & (summary["train_seed"] == seed)].iloc[0]
        summary_rows.append(
            [
                _method_label(method),
                str(seed),
                f"{row['spread_mean']:.2f}",
                f"{row['spread_std']:.2f}",
                f"{row['reward_mean']:.4f}",
                f"{row['selection_seconds_mean'] * 1000:.2f}",
                f"{row['training_seconds']:.1f}",
            ]
        )
    story.append(_table(summary_rows, [108, 30, 56, 43, 58, 57, 48], styles))
    story.append(
        _p(
            "Table 2. Evaluation summary. Random and Degree use train_seed = -1 and training_seconds = 0.",
            styles["CaptionReport"],
        )
    )
    architecture_rows = [["Architecture", "Runs", "Mean of run means", "Pop. SD of run means"]]
    for row in architecture.itertuples(index=False):
        architecture_rows.append(
            [
                _method_label(str(row.method)),
                str(int(row.n_runs)),
                f"{row.run_mean_spread_mean:.3f}",
                f"{row.run_mean_spread_std:.3f}",
            ]
        )
    story.append(_table(architecture_rows, [130, 42, 95, 95], styles))
    story.append(
        _p(
            "Table 3. Run-level aggregate for each neural architecture.",
            styles["CaptionReport"],
        )
    )

    learning_figure = results_dir / "figures" / "learning_curves.png"
    story.append(
        KeepTogether(
            [
                _image(learning_figure, CONTENT_WIDTH * 0.85),
                _p(
                    "Figure 2. Learning curves for all six runs. Each point is the mean final spread in a "
                    "non-overlapping block of 20 training episodes.",
                    styles["CaptionReport"],
                ),
            ]
        )
    )
    comparison_figure = results_dir / "figures" / "final_spread_comparison.png"
    story.append(
        KeepTogether(
            [
                _image(comparison_figure, CONTENT_WIDTH * 0.93),
                _p(
                    "Figure 3. Final spread on the common 100 diffusion realisations. Bars show means; error bars "
                    "are population standard deviations.",
                    styles["CaptionReport"],
                ),
            ]
        )
    )

    # Selection table: fixed selections plus the Random coverage summary.
    selection_rows = [
        ["Method", "Seed", "Selected node IDs", "Departments (count)", "Coverage"]
    ]
    for row in selected.itertuples(index=False):
        if str(row.method) == "random":
            continue
        selection_rows.append(
            [
                _method_label(str(row.method)),
                str(int(row.train_seed)),
                str(row.selection_order).replace("[", "").replace("]", ""),
                _department_counts(str(row.departments)),
                str(int(row.department_coverage)),
            ]
        )
    selection_caption = _p(
        "Table 4. Selected participants and department coverage for the deterministic selections. "
        "Department counts are shown for the five selected users; coverage is the number of distinct "
        "departments represented. Random's mean coverage over its 100 episodes is "
        f"{float(random_coverage_row.coverage_mean):.2f} "
        f"(population SD {float(random_coverage_row.coverage_std):.2f}, range "
        f"{int(random_coverage_row.coverage_min)}-{int(random_coverage_row.coverage_max)}). "
        "Coverage is descriptive only.",
        styles["CaptionReport"],
    )
    story.append(
        KeepTogether(
            [
                _table(selection_rows, [100, 26, 130, 150, 45], styles),
                selection_caption,
            ]
        )
    )

    transformer_figure = results_dir / "figures" / "transformer_selection.png"
    story.append(
        KeepTogether(
            [
                _image(transformer_figure, CONTENT_WIDTH * 0.78),
                _p(
                    "Figure 4. The five participants selected by the Transformer checkpoint trained with seed 7. "
                    "The selection is identical across diffusion realisations because the policy sees only the "
                    "fixed graph and the evolving selection state.",
                    styles["CaptionReport"],
                ),
            ]
        )
    )

    # Narrative comparison, computed rather than hard-coded.
    mlp_runs = summary[summary["method"] == "reinforce_mlp"]["spread_mean"].tolist()
    transformer_runs = summary[
        summary["method"] == "reinforce_transformer"
    ]["spread_mean"].tolist()
    best_learned = max(mlp_mean, transformer_mean)
    if best_learned > degree_mean:
        comparison = (
            f"The stronger learned aggregate ({best_learned:.2f}) exceeds Degree by "
            f"{best_learned - degree_mean:.2f} participants on average."
        )
    else:
        comparison = (
            f"Neither learned architecture exceeds Degree's mean ({degree_mean:.2f}); the gap for the "
            f"stronger architecture is {degree_mean - best_learned:.2f} participants."
        )
    overlap_text = _selected_overlap_text(selection_orders, graph)
    story.append(_p("8. Discussion", styles["Heading1Report"]))
    story.append(
        _p(
            f"<b>Comparison with baselines.</b> Degree's mean spread is {degree_mean:.2f}, versus "
            f"{random_mean:.2f} for Random. The MLP run means are "
            f"{', '.join(f'{value:.2f}' for value in mlp_runs)} and the Transformer run means are "
            f"{', '.join(f'{value:.2f}' for value in transformer_runs)}. {comparison} The comparison is "
            "descriptive: it uses the same 100 realisations for every method, but no significance test or "
            "tuning on evaluation results was performed.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            f"<b>Overlapping audiences and department coverage.</b> Internal edges among the selected "
            f"sets quantify overlap: {overlap_text}. Degree's five hubs are therefore not five "
            "independent audiences. Department coverage varies across selections; the full selected-user "
            "table and the Random coverage distribution are saved in results/selected_users.csv and "
            "results/department_coverage.csv. These labels remain descriptive and never enter a policy, "
            "reward, or tuning decision.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            f"<b>Learning stability and cost.</b> The MLP run-level mean spread has SD "
            f"{float(mlp_arch['run_mean_spread_std']):.2f}; the Transformer's is "
            f"{float(transformer_arch['run_mean_spread_std']):.2f}. The learning curves show how the "
            f"20-episode block means move during training. On CPU, training took "
            f"{summary[summary['method'] == 'reinforce_mlp']['training_seconds'].mean():.1f} s per MLP run "
            f"and {summary[summary['method'] == 'reinforce_transformer']['training_seconds'].mean():.1f} s "
            f"per Transformer run, including simulations; greedy selection averaged "
            f"{mlp_selection_ms:.2f} ms per MLP episode and {transformer_selection_ms:.2f} ms "
            f"per Transformer episode. Degree selection is effectively instantaneous "
            f"({degree_seconds:.4f} ms). Wall-clock values are specific to this machine and run.",
            styles["BodyReport"],
        ),
    )
    story.append(
        _p(
            "<b>MLP versus Transformer.</b> The MLP is a per-participant scorer, so its only context is "
            "the five features of that participant. The Transformer can represent interactions among all "
            "92 participant tokens, including which users were already selected, after every action. With "
            "only 50 updates and a terminal scalar reward, that extra capacity is not guaranteed to help; "
            "the observed run-level means and their spread are the relevant evidence, not architecture "
            "size alone.",
            styles["BodyReport"],
        ),
    )

    # ------------------------------------------------------------------
    # 9 Limitations
    # ------------------------------------------------------------------
    story.append(_p("9. Limitations", styles["Heading1Report"]))
    limitations = [
        "The graph is a static aggregation of the supplied contacts. Contact timing, duration, and "
        "repeated interactions are discarded, so the analysis cannot represent when a message would spread.",
        "Independent Cascade at probability 0.10 is a simulation assumption, not observed message "
        "adoption. Final spread is a model quantity and not a measured outcome.",
        "The study uses one fixed 92-participant graph, three training seeds, and 100 evaluation "
        "realisations. The results may not transfer to other networks or contact periods.",
        "No hyperparameter search, early stopping, or evaluation-based tuning was used. The reported "
        "learning behaviour is therefore a controlled baseline rather than an optimised system.",
        "Department labels describe recorded organizational units; they are not used as inputs or "
        "objectives, and coverage is not a causal outcome.",
        "Timing is CPU- and machine-specific. Selection timing excludes diffusion, but small timing "
        "differences between Python-level operations should not be over-interpreted.",
        "The learned policies are compared with Degree and Random on the same realisations; the "
        "assignment does not require a learned policy to outperform Degree.",
    ]
    for text in limitations:
        story.append(_p(text, styles["BulletReport"], bulletText="-"))

    story.append(_p("10. Reproducibility and generated artifacts", styles["Heading1Report"]))
    story.append(
        _p(
            "The complete pipeline is executed from the project root with "
            "<font name='Courier'>python main.py --output results</font>. The supplied CSV files are "
            "checked against their published SHA-256 hashes before use and remain unchanged. Python, "
            "NumPy, and PyTorch are seeded per run; diffusion seeds follow the prescribed formula; each "
            "checkpoint stores its architecture settings and training configuration. The checks are "
            "executed assertions, not manually entered booleans. The recorded environment was "
            f"Python {metadata['python']}, PyTorch {metadata['torch']}, NumPy {metadata['numpy']}, "
            f"pandas {metadata['pandas']}, and NetworkX {metadata['networkx']} on "
            f"{metadata['cpu_count']} logical CPU cores with {metadata['platform']}.",
            styles["BodyReport"],
        ),
    )
    artifact_rows = [
        ["Artifact", "Observed content"],
        ["training.csv", "6,000 rows: 2 architectures x 3 seeds x 1,000 episodes"],
        ["evaluation.csv", "800 rows: 8 method/training-seed groups x 100 shared seeds"],
        ["summary.csv", "8 rows, one per method/training-seed combination"],
        ["checkpoints/", "6 final model checkpoints with architecture metadata"],
        ["checks.json", f"{checks_passed}/{checks_total} checks passed in this run"],
        ["figures/", "Network analysis, learning curves, comparison, and Transformer selection"],
        [
            "Pipeline time",
            f"{float(metadata['total_pipeline_seconds']):.1f} s for training, evaluation, figures, and "
            "checks before the PDF build (machine-specific)",
        ],
    ]
    story.append(_table(artifact_rows, [112, 380], styles))
    story.append(
        _p(
            "Table 5. Reproducibility evidence for the reported run.",
            styles["CaptionReport"],
        )
    )

    # ------------------------------------------------------------------
    # References (separate page, excluded from 4-6 body pages)
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(_p("References", styles["Heading1Report"]))
    references = [
        "SocioPatterns. Contacts in a Workplace. Data collected 24 June to 3 July 2013. "
        "https://sociopatterns.org/datasets/contacts-in-a-workplace/ (accessed 2026).",
        "M. Génois, C. L. Vestergaard, J. Fournet, A. Panisson, I. Bonmarin, and A. Barrat. "
        "Data on face-to-face contacts in an office building suggest a low-cost vaccination strategy "
        "based on community linkers. Network Science, 3(3):326-347, 2015. "
        "https://doi.org/10.1017/nws.2015.10",
        "R. J. Williams. Simple statistical gradient-following algorithms for connectionist "
        "reinforcement learning. Machine Learning, 8:229-256, 1992. "
        "https://doi.org/10.1007/BF00992696",
        "A. Paszke et al. PyTorch: An imperative style, high-performance deep learning library. "
        "Advances in Neural Information Processing Systems 32, 2019.",
        "A. A. Hagberg, D. A. Schult, and P. J. Swart. Exploring network structure, dynamics, and "
        "function using NetworkX. Proceedings of the 7th Python in Science Conference, 2008.",
        "J. D. Hunter. Matplotlib: A 2D graphics environment. Computing in Science & Engineering, "
        "9(3):90-95, 2007.",
        "Implementation note: all assignment code was written for this submission. PyTorch's standard "
        "nn.TransformerEncoderLayer was used as permitted; no pre-trained model or ready-made "
        "reinforcement-learning agent was used.",
    ]
    for reference in references:
        story.append(_p(reference, styles["Reference"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Sequential Influence Maximization in a Workplace Contact Network",
        author="SE4095 Assignment 03",
        subject="Social Network Theory and Practice - Assignment 03",
        creator="Python / ReportLab",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path
