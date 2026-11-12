from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from textwrap import wrap
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np


PALETTE = {
    "ink": "#1f2937",
    "muted": "#6b7280",
    "grid": "#d1d5db",
    "paper": "#ffffff",
    "traffic": "#3b82f6",
    "protocol": "#10b981",
    "temporal": "#f59e0b",
    "fusion": "#8b5cf6",
    "alert": "#ef4444",
    "agent": "#0f766e",
    "slate": "#475569",
    "soft_blue": "#dbeafe",
    "soft_green": "#d1fae5",
    "soft_amber": "#fef3c7",
    "soft_purple": "#ede9fe",
    "soft_red": "#fee2e2",
    "soft_gray": "#f3f4f6",
}

FIGURE_SPECS = [
    ("fig_01_atlas_pipeline_xfig", "ATLAS-IDNet end-to-end detection and evidence pipeline."),
    ("fig_02_multibranch_layers_xfig", "ATLAS-IDNet multi-branch neural architecture."),
    ("fig_03_agentic_explainability_xfig", "Evidence-grounded explanation-agent workflow."),
    ("fig_04_related_work_numeric_summary_xfig", "XFig-style evidence graph of the reviewed IDS journal literature."),
    ("fig_05_main_metrics_heatmap_xfig", "Dataset-wise ATLAS-IDNet metric heatmap."),
    ("fig_06_ablation_delta_xfig", "Branch-masking ablation effect on weighted F1-score."),
    ("fig_07_latency_realtime_xfig", "Real-time inference latency and throughput."),
    ("fig_08_attention_summary_xfig", "Learned branch-attention summary."),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | int | float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _figure(width: float = 12.0, height: float = 7.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, output_dir: Path, stem: str, png_dpi: int = 320) -> list[str]:
    paths = []
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
    paths.append(str(png_path))
    for ext in ("svg", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _label(ax, x: float, y: float, text: str, size: float = 9.0, weight: str = "normal", color: str | None = None,
           ha: str = "center", va: str = "center", max_chars: int = 22, zorder: int = 5) -> None:
    lines = "\n".join(wrap(text, width=max_chars, break_long_words=False))
    ax.text(x, y, lines, ha=ha, va=va, fontsize=size, fontweight=weight, color=color or PALETTE["ink"], zorder=zorder)


def _box(ax, xy: tuple[float, float], wh: tuple[float, float], text: str, face: str, edge: str | None = None,
         size: float = 9.0, weight: str = "normal", max_chars: int = 22) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge or PALETTE["ink"], linewidth=1.2, zorder=2))
    _label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, max_chars=max_chars)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str | None = None, lw: float = 1.4,
           rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color or PALETTE["ink"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            zorder=3,
        )
    )


def _node(ax, x: float, y: float, radius: float, face: str, edge: str | None = None) -> None:
    ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge or PALETTE["ink"], linewidth=1.0, zorder=3))


def _title(ax, text: str, subtitle: str | None = None) -> None:
    _label(ax, 0.02, 0.965, text, size=13, weight="bold", ha="left", max_chars=90)
    if subtitle:
        _label(ax, 0.02, 0.925, subtitle, size=8.5, color=PALETTE["muted"], ha="left", max_chars=120)
    ax.plot([0.02, 0.98], [0.895, 0.895], color=PALETTE["grid"], linewidth=1.0)


def _metric_heatmap(ax, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], x0: float, y0: float,
                    width: float, height: float, cmap: str = "YlGnBu", value_format: str = "{:.2f}") -> None:
    rows, cols = matrix.shape
    cell_w = width / max(cols, 1)
    cell_h = height / max(rows, 1)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    colors = plt.get_cmap(cmap)
    for r in range(rows):
        for c in range(cols):
            value = matrix[r, c]
            norm = 0.0 if not np.isfinite(value) else (float(value) - vmin) / (vmax - vmin)
            ax.add_patch(
                Rectangle(
                    (x0 + c * cell_w, y0 + (rows - r - 1) * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=colors(norm),
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            if np.isfinite(value):
                _label(
                    ax,
                    x0 + c * cell_w + cell_w / 2,
                    y0 + (rows - r - 1) * cell_h + cell_h / 2,
                    value_format.format(value),
                    size=8,
                    color=PALETTE["ink"],
                    max_chars=12,
                )
    for c, label in enumerate(col_labels):
        _label(ax, x0 + c * cell_w + cell_w / 2, y0 + height + 0.028, label, size=8, weight="bold", max_chars=13)
    for r, label in enumerate(row_labels):
        _label(ax, x0 - 0.018, y0 + (rows - r - 1) * cell_h + cell_h / 2, label, size=8.5, ha="right", max_chars=18)


def draw_atlas_pipeline(output_dir: Path) -> dict:
    stem = "fig_01_atlas_pipeline_xfig"
    fig, ax = _figure(13.2, 7.0)
    _title(ax, "ATLAS-IDNet Detection Pipeline", "Data preparation, multi-branch detection, evidence extraction, and reporting are separated for reproducible IDS evaluation.")

    left_x = 0.05
    stages = [
        ("CICIoT2023\nNF-UQ-NIDS-v2", PALETTE["soft_blue"]),
        ("Schema audit\nlabel mapping", PALETTE["soft_gray"]),
        ("Cleaning\nencoding\nscaling", PALETTE["soft_green"]),
        ("Stratified\ntrain/val/test", PALETTE["soft_amber"]),
    ]
    x = left_x
    y = 0.70
    for i, (text, face) in enumerate(stages):
        _box(ax, (x, y), (0.13, 0.11), text, face, size=8.5, weight="bold", max_chars=14)
        if i < len(stages) - 1:
            _arrow(ax, (x + 0.13, y + 0.055), (x + 0.155, y + 0.055), color=PALETTE["slate"])
        x += 0.16

    branch_x = 0.66
    branch_specs = [
        ("Traffic branch\nbytes, packets, rates", PALETTE["soft_blue"], PALETTE["traffic"], 0.79),
        ("Protocol branch\nports, flags, services", PALETTE["soft_green"], PALETTE["protocol"], 0.63),
        ("Temporal branch\nduration, flow stats", PALETTE["soft_amber"], PALETTE["temporal"], 0.47),
    ]
    for text, face, edge, by in branch_specs:
        _box(ax, (branch_x, by - 0.045), (0.145, 0.09), text, face, edge=edge, size=7.6, weight="bold", max_chars=16)
        _arrow(ax, (0.53, 0.755), (branch_x, by), color=edge, rad=0.05 if by != 0.63 else 0.0)

    _box(ax, (0.815, 0.57), (0.105, 0.12), "Attention\nfusion", PALETTE["soft_purple"], edge=PALETTE["fusion"], size=7.2, weight="bold", max_chars=10)
    for _, _, edge, by in branch_specs:
        _arrow(ax, (branch_x + 0.145, by), (0.815, 0.63), color=edge, rad=0.08 if by != 0.63 else 0.0)
    _box(ax, (0.943, 0.57), (0.052, 0.12), "Attack\nclass", PALETTE["soft_red"], edge=PALETTE["alert"], size=6.4, weight="bold", max_chars=7)
    _arrow(ax, (0.92, 0.63), (0.943, 0.63), color=PALETTE["alert"])

    evidence = [
        ("Metrics\nF1, MCC, FAR", 0.16, PALETTE["soft_blue"]),
        ("Ablation\nbranch masking", 0.36, PALETTE["soft_green"]),
        ("Latency\nms/sample", 0.56, PALETTE["soft_amber"]),
        ("Explainability\nfeature and branch evidence", 0.76, PALETTE["soft_purple"]),
    ]
    for text, ex, face in evidence:
        _box(ax, (ex - 0.075, 0.18), (0.15, 0.11), text, face, size=8, weight="bold", max_chars=17)
    ax.plot([0.085, 0.835], [0.335, 0.335], color=PALETTE["grid"], linewidth=1.0)
    _label(ax, 0.46, 0.355, "saved evaluation artifacts", size=8, color=PALETTE["muted"], max_chars=34)
    _box(ax, (0.41, 0.035), (0.22, 0.08), "Evidence-grounded agentic report", PALETTE["soft_gray"], edge=PALETTE["agent"], size=9, weight="bold", max_chars=28)
    for _, ex, _ in evidence:
        _arrow(ax, (ex, 0.18), (0.52, 0.115), color=PALETTE["agent"], rad=0.05)

    return {"caption": FIGURE_SPECS[0][1], "files": _save(fig, output_dir, stem)}


def draw_multibranch_layers(output_dir: Path) -> dict:
    stem = "fig_02_multibranch_layers_xfig"
    fig, ax = _figure(13.2, 7.0)
    _title(ax, "Multi-Branch Deep IDS Architecture", "The detector separates heterogeneous flow features before attention-weighted representation fusion.")

    x_positions = [0.08, 0.23, 0.36]
    labels = ["Input\nfeature vector", "Feature\npartition", "Branch\nencoders"]
    for x, label in zip(x_positions, labels):
        _box(ax, (x - 0.055, 0.70), (0.11, 0.10), label, PALETTE["soft_gray"], size=8.3, weight="bold", max_chars=14)
    _arrow(ax, (0.135, 0.75), (0.175, 0.75), color=PALETTE["slate"])
    _arrow(ax, (0.285, 0.75), (0.325, 0.75), color=PALETTE["slate"])

    branch_ys = [0.79, 0.63, 0.47]
    branch_names = ["Traffic", "Protocol", "Temporal"]
    branch_colors = [PALETTE["traffic"], PALETTE["protocol"], PALETTE["temporal"]]
    branch_faces = [PALETTE["soft_blue"], PALETTE["soft_green"], PALETTE["soft_amber"]]
    for y, name, color, face in zip(branch_ys, branch_names, branch_colors, branch_faces):
        _box(ax, (0.46, y - 0.04), (0.095, 0.08), f"{name}\nfeatures", face, edge=color, size=7.4, weight="bold", max_chars=11)
        previous = 0.555
        for i, layer in enumerate(["Dense", "BN+GELU", "Residual", "Residual"]):
            _box(ax, (previous + i * 0.058, y - 0.035), (0.048, 0.07), layer, "white", edge=color, size=6.0, max_chars=8)
            if i > 0:
                _arrow(ax, (previous + i * 0.058 - 0.010, y), (previous + i * 0.058, y), color=color, lw=1.0)

    _box(ax, (0.79, 0.57), (0.10, 0.14), "Softmax\nbranch\nattention", PALETTE["soft_purple"], edge=PALETTE["fusion"], size=8, weight="bold", max_chars=12)
    for y, color in zip(branch_ys, branch_colors):
        _arrow(ax, (0.778, y), (0.79, 0.64), color=color, rad=0.08 if y != 0.63 else 0.0)

    _box(ax, (0.91, 0.57), (0.075, 0.14), "Fused\nstate", PALETTE["soft_gray"], edge=PALETTE["ink"], size=8, weight="bold")
    _arrow(ax, (0.89, 0.64), (0.91, 0.64), color=PALETTE["fusion"])
    _box(ax, (0.83, 0.32), (0.13, 0.08), "Classifier\nresidual block", "white", edge=PALETTE["ink"], size=8, weight="bold", max_chars=16)
    _arrow(ax, (0.945, 0.57), (0.895, 0.40), color=PALETTE["ink"], rad=-0.15)
    _box(ax, (0.80, 0.17), (0.20, 0.08), "Normal / attack-family prediction", PALETTE["soft_red"], edge=PALETTE["alert"], size=8.3, weight="bold", max_chars=24)
    _arrow(ax, (0.895, 0.32), (0.90, 0.25), color=PALETTE["alert"])

    for i in range(5):
        _node(ax, 0.09, 0.44 + i * 0.045, 0.012, PALETTE["soft_blue"], PALETTE["traffic"])
        _node(ax, 0.16, 0.42 + i * 0.055, 0.012, PALETTE["soft_green"], PALETTE["protocol"])
        _arrow(ax, (0.102, 0.44 + i * 0.045), (0.148, 0.42 + i * 0.055), color=PALETTE["grid"], lw=0.8)
    _label(ax, 0.125, 0.36, "Layered feature abstraction", size=8, color=PALETTE["muted"], max_chars=24)

    return {"caption": FIGURE_SPECS[1][1], "files": _save(fig, output_dir, stem)}


def draw_agentic_explainability(output_dir: Path) -> dict:
    stem = "fig_03_agentic_explainability_xfig"
    fig, ax = _figure(13.2, 7.0)
    _title(ax, "Evidence-Grounded Explanation Agent", "The reporting agent reads saved artifacts and emits constrained explanations tied to detector evidence.")

    sources = [
        ("EDA artifacts", 0.08, 0.74, PALETTE["soft_blue"]),
        ("Test metrics", 0.08, 0.59, PALETTE["soft_green"]),
        ("Ablation table", 0.08, 0.44, PALETTE["soft_amber"]),
        ("Latency profile", 0.08, 0.29, PALETTE["soft_red"]),
        ("Attribution files", 0.08, 0.14, PALETTE["soft_purple"]),
    ]
    for text, x, y, face in sources:
        _box(ax, (x, y), (0.16, 0.09), text, face, size=8.4, weight="bold", max_chars=17)
        _arrow(ax, (x + 0.16, y + 0.045), (0.36, 0.50), color=PALETTE["slate"], rad=0.12)

    _box(ax, (0.36, 0.39), (0.20, 0.22), "Evidence router\n\nvalidates available files\nseparates missing evidence\nnormalizes metric language", "white", edge=PALETTE["agent"], size=8.2, weight="bold", max_chars=25)
    _box(ax, (0.64, 0.52), (0.16, 0.11), "Structured JSON\nclaims", PALETTE["soft_gray"], edge=PALETTE["ink"], size=8.5, weight="bold")
    _box(ax, (0.64, 0.35), (0.16, 0.11), "Markdown analyst\nreport", PALETTE["soft_gray"], edge=PALETTE["ink"], size=8.5, weight="bold")
    _arrow(ax, (0.56, 0.50), (0.64, 0.575), color=PALETTE["agent"])
    _arrow(ax, (0.56, 0.50), (0.64, 0.405), color=PALETTE["agent"])

    _box(ax, (0.86, 0.44), (0.11, 0.10), "Paper-ready\nstatements", PALETTE["soft_green"], edge=PALETTE["protocol"], size=8, weight="bold", max_chars=14)
    _arrow(ax, (0.80, 0.575), (0.86, 0.50), color=PALETTE["protocol"], rad=-0.08)
    _arrow(ax, (0.80, 0.405), (0.86, 0.49), color=PALETTE["protocol"], rad=0.08)

    checks = [
        ("No unsupported metric", 0.27),
        ("No surrogate substitution", 0.18),
        ("No hidden retraining claim", 0.09),
    ]
    for text, y in checks:
        _node(ax, 0.64, y, 0.012, PALETTE["soft_green"], PALETTE["protocol"])
        _label(ax, 0.665, y, text, size=8, ha="left", color=PALETTE["ink"], max_chars=26)

    return {"caption": FIGURE_SPECS[2][1], "files": _save(fig, output_dir, stem)}


def _theme_for(row: dict[str, str]) -> str:
    text = " ".join([row.get("title", ""), row.get("relevance_to_atlas_idnet", "")]).lower()
    if any(token in text for token in ["explainable", "xai", "feature importance"]):
        return "Explainability"
    if any(token in text for token in ["federated", "transfer"]):
        return "Federated"
    if any(token in text for token in ["edge", "tinyml", "lightweight", "uav", "manet", "kubernetes", "real-time"]):
        return "Edge/Real-time"
    if any(token in text for token in ["transformer", "attention", "graph attention"]):
        return "Attention/Transformer"
    if any(token in text for token in ["deep", "cnn", "lstm", "trbm", "hybrididsnet", "autoencoder"]):
        return "Deep models"
    if any(token in text for token in ["boosting", "ensemble", "knn", "machine learning", "random forest"]):
        return "ML/Ensemble"
    return "IDS benchmark"


def draw_literature_numeric_summary(literature_csv: Path, output_dir: Path) -> dict:
    stem = "fig_04_related_work_numeric_summary_xfig"
    rows = [row for row in _read_csv(literature_csv) if row.get("status") == "downloaded"]
    years = ["2023", "2024", "2025", "2026"]
    themes = ["ML/Ensemble", "Deep models", "Attention/Transformer", "Federated", "Explainability", "Edge/Real-time", "IDS benchmark"]
    year_counts = Counter(str(row.get("year_requested", "")) for row in rows)
    method_rows = [row for row in rows if row.get("key") != "neto2023ciciot2023"]
    theme_counts = Counter(_theme_for(row) for row in method_rows)
    total = len(rows)
    dataset_count = sum(1 for row in rows if row.get("key") == "neto2023ciciot2023")
    method_count = total - dataset_count

    fig, ax = _figure(16.5, 8.2)
    ax.set_ylim(-0.14, 1.0)
    _title(
        ax,
        "Related Work Evidence Graph",
        "XFig-style count diagram derived from the verified 2023--2026 journal-paper matrix.",
    )

    mono = "#111827"
    edge = "#374151"
    lite = "#f8fafc"
    gate = "#f1f5f9"

    def diagram_box(x: float, y: float, w: float, h: float, text: str, face: str = "white",
                    line: str = edge, size: float = 8.0, weight: str = "normal",
                    chars: int = 24, hatch: str | None = None) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=face,
                edgecolor=line,
                linewidth=1.05,
                hatch=hatch,
                zorder=2,
            )
        )
        _label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, max_chars=chars)

    def joint_arrow(start: tuple[float, float], mid: tuple[float, float], end: tuple[float, float],
                    color: str = mono) -> None:
        ax.plot([start[0], mid[0]], [start[1], mid[1]], color=color, linewidth=1.05, zorder=1)
        ax.add_patch(
            FancyArrowPatch(
                mid,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.05,
                color=color,
                shrinkA=0,
                shrinkB=4,
                zorder=3,
            )
        )

    def badge(x: float, y: float, value: int | str, face: str = "white") -> None:
        _node(ax, x, y, 0.020, face, mono)
        _label(ax, x, y, str(value), size=7.4, weight="bold", max_chars=3)

    diagram_box(0.37, 0.77, 0.26, 0.10, f"Verified IDS journal corpus\nN = {total}", lite, mono, 9.5, "bold", 28)
    diagram_box(0.07, 0.77, 0.22, 0.10, "Selection basis\n2023--2026, journal-only\npublished version", gate, edge, 7.5, "bold", 26)
    diagram_box(0.71, 0.77, 0.22, 0.10, "Quality gate\nstable DOI, publisher page\ndownloadable PDF", gate, edge, 7.5, "bold", 26)
    joint_arrow((0.29, 0.82), (0.34, 0.82), (0.37, 0.82))
    joint_arrow((0.63, 0.82), (0.68, 0.82), (0.71, 0.82))

    diagram_box(0.18, 0.61, 0.23, 0.09, f"Method papers\nN = {method_count}", PALETTE["soft_blue"], mono, 8.6, "bold", 24)
    diagram_box(0.59, 0.61, 0.23, 0.09, f"Dataset benchmark source\nN = {dataset_count}", PALETTE["soft_amber"], mono, 8.6, "bold", 25)
    joint_arrow((0.50, 0.77), (0.50, 0.735), (0.30, 0.705))
    joint_arrow((0.50, 0.77), (0.50, 0.735), (0.70, 0.705))

    _label(ax, 0.05, 0.535, "A. publication-year count nodes", size=8.6, weight="bold", ha="left", max_chars=42)
    year_nodes = [("2023", 0.07), ("2024", 0.19), ("2025", 0.31), ("2026", 0.43)]
    for year, x in year_nodes:
        diagram_box(x, 0.43, 0.085, 0.065, year, "white", mono, 8.0, "bold", 10)
        badge(x + 0.043, 0.405, year_counts[year], PALETTE["soft_blue"])
        joint_arrow((0.295, 0.61), (0.295, 0.54), (x + 0.043, 0.498))

    _label(ax, 0.56, 0.535, "B. method-stream count nodes", size=8.6, weight="bold", ha="left", max_chars=42)
    stream_order = [
        ("Deep models", "Deep / hybrid DL"),
        ("ML/Ensemble", "ML / ensemble"),
        ("Edge/Real-time", "Edge / real-time"),
        ("Attention/Transformer", "Attention / transformer"),
        ("IDS benchmark", "Benchmark / evaluation"),
        ("Federated", "Federated IDS"),
        ("Explainability", "Explainability"),
    ]
    stream_positions = [(0.63, y) for y in [0.475, 0.415, 0.355, 0.295, 0.235, 0.175, 0.115]]
    bus_x = 0.56
    ax.plot([0.41, bus_x], [0.655, 0.655], color=mono, linewidth=1.05, zorder=1)
    ax.plot([bus_x, bus_x], [0.655, 0.145], color=mono, linewidth=1.05, zorder=1)
    for (theme, display), (x, y) in zip(stream_order, stream_positions):
        count = theme_counts[theme]
        diagram_box(x, y, 0.185, 0.045, display, "white", edge, 6.4, "bold", 24)
        badge(x + 0.200, y + 0.023, count, PALETTE["soft_gray"])
        joint_arrow((bus_x, y + 0.023), (0.60, y + 0.023), (x, y + 0.023))

    ax.plot([0.05, 0.95], [0.075, 0.075], color=mono, linewidth=0.85)
    _label(ax, 0.05, 0.045, "C. gap-to-positioning graph", size=8.6, weight="bold", ha="left", max_chars=42)
    gap_items = [
        ("Accuracy-centric\nreporting", 0.08),
        ("Limited latency\nevidence", 0.25),
        ("Sparse model\nablation", 0.42),
        ("Weak explanation\nprovenance", 0.59),
    ]
    for text, x in gap_items:
        diagram_box(x, -0.065, 0.12, 0.075, text, "#fff7ed", mono, 6.7, "bold", 16, hatch="..")
        joint_arrow((x + 0.06, 0.075), (x + 0.06, 0.045), (x + 0.06, 0.010))
    diagram_box(0.78, -0.090, 0.17, 0.125, "ATLAS-IDNet position\nmulti-branch detection\nmetrics, latency, ablation,\nexplanation evidence", PALETTE["soft_green"], mono, 6.1, "bold", 22)
    for _, x in gap_items:
        joint_arrow((x + 0.12, -0.028), (0.73, -0.028), (0.78, -0.028))

    _label(
        ax,
        0.50,
        -0.110,
        "The graph positions the proposed study against the verified journal corpus without using conference, preprint, or early-access records.",
        size=7.6,
        color=PALETTE["muted"],
        max_chars=128,
    )
    return {"caption": FIGURE_SPECS[3][1], "files": _save(fig, output_dir, stem, png_dpi=1600), "source": str(literature_csv)}


def draw_main_metrics(summary_dir: Path, output_dir: Path) -> dict | None:
    rows = _read_csv(summary_dir / "table_main_atlas_metrics.csv")
    if not rows:
        return None
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "matthews_correlation_coefficient"]
    labels = ["Acc.", "Prec.", "Recall", "W-F1", "M-F1", "MCC"]
    datasets = [row.get("dataset", "dataset") for row in rows]
    matrix = np.array([[_float(row.get(metric), np.nan) for metric in metrics] for row in rows], dtype=float)

    fig, ax = _figure(13.2, 7.0)
    _title(ax, "ATLAS-IDNet Metric Heatmap", "Dataset-wise standard metrics are loaded from the generated paper-summary artifacts.")
    _metric_heatmap(ax, matrix, datasets, labels, 0.24, 0.30, 0.62, 0.34, cmap="YlGnBu", value_format="{:.3f}")
    _label(ax, 0.24, 0.22, "Higher values indicate stronger detection quality.", size=8, ha="left", color=PALETTE["muted"], max_chars=60)
    return {"caption": FIGURE_SPECS[4][1], "files": _save(fig, output_dir, "fig_05_main_metrics_heatmap_xfig"), "source": str(summary_dir / "table_main_atlas_metrics.csv")}


def draw_ablation(summary_dir: Path, output_dir: Path) -> dict | None:
    rows = _read_csv(summary_dir / "table_atlas_branch_ablation.csv")
    rows = [row for row in rows if row.get("variant") != "full_atlas"]
    if not rows:
        return None
    labels = [f"{row.get('dataset', '')}\n{row.get('variant', '')}".replace("_", " ") for row in rows]
    values = np.array([_float(row.get("delta_f1_weighted_vs_full_atlas")) for row in rows])
    order = np.argsort(values)
    values = values[order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(12.5, max(5.0, 0.42 * len(labels))), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = [PALETTE["protocol"] if v >= 0 else PALETTE["alert"] for v in values]
    ax.barh(range(len(labels)), values, color=colors, edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.axvline(0, color=PALETTE["ink"], linewidth=1.0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Delta weighted F1-score versus full ATLAS-IDNet")
    ax.set_title("Branch-Masking Ablation")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return {"caption": FIGURE_SPECS[5][1], "files": _save(fig, output_dir, "fig_06_ablation_delta_xfig"), "source": str(summary_dir / "table_atlas_branch_ablation.csv")}


def draw_latency(summary_dir: Path, output_dir: Path) -> dict | None:
    rows = _read_csv(summary_dir / "table_latency_realtime.csv")
    if not rows:
        return None
    labels = [row.get("dataset", "dataset") for row in rows]
    latency = np.array([_float(row.get("latency_ms_per_sample")) for row in rows])
    throughput = np.array([_float(row.get("throughput_samples_per_second")) for row in rows])

    fig, ax1 = plt.subplots(figsize=(10.5, 5.8), dpi=220)
    fig.patch.set_facecolor("white")
    x = np.arange(len(labels))
    ax1.bar(x - 0.18, latency, width=0.36, color=PALETTE["traffic"], edgecolor=PALETTE["ink"], linewidth=0.6, label="Latency (ms/sample)")
    ax1.set_ylabel("Latency (ms/sample)", color=PALETTE["traffic"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["traffic"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, throughput, color=PALETTE["agent"], marker="s", linewidth=1.8, label="Throughput")
    ax2.set_ylabel("Throughput (samples/s)", color=PALETTE["agent"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["agent"])
    ax1.set_title("Real-Time Inference Profile")
    for spine in ["top"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    fig.tight_layout()
    return {"caption": FIGURE_SPECS[6][1], "files": _save(fig, output_dir, "fig_07_latency_realtime_xfig"), "source": str(summary_dir / "table_latency_realtime.csv")}


def draw_attention(summary_dir: Path, output_dir: Path) -> dict | None:
    rows = _read_csv(summary_dir / "table_attention_summary.csv")
    if not rows:
        return None
    branch_key = "attention_branch"
    if branch_key not in rows[0]:
        candidates = [key for key in rows[0].keys() if key not in {"dataset", "mean", "std", "min", "50%", "max"}]
        branch_key = candidates[0] if candidates else "attention_branch"
    labels = [f"{row.get('dataset', '')}\n{row.get(branch_key, '')}" for row in rows]
    means = np.array([_float(row.get("mean")) for row in rows])
    errors = np.array([_float(row.get("std")) for row in rows])

    fig, ax = plt.subplots(figsize=(11.5, max(5.0, 0.38 * len(labels))), dpi=220)
    fig.patch.set_facecolor("white")
    ax.barh(range(len(labels)), means, xerr=errors, color=PALETTE["fusion"], edgecolor=PALETTE["ink"], linewidth=0.5, ecolor=PALETTE["muted"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean attention weight")
    ax.set_xlim(0, max(1.0, float(np.nanmax(means + errors)) * 1.08))
    ax.set_title("ATLAS-IDNet Branch-Attention Summary")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return {"caption": FIGURE_SPECS[7][1], "files": _save(fig, output_dir, "fig_08_attention_summary_xfig"), "source": str(summary_dir / "table_attention_summary.csv")}


def _latex_snippets(outputs: dict[str, dict], output_dir: Path) -> str:
    lines = [
        "% Auto-generated figure snippets for IEEE Access manuscript integration.",
        "% Verify placement and captions before final submission.",
        "",
    ]
    for stem, item in outputs.items():
        figure_file = next((Path(path).name for path in item.get("files", []) if str(path).endswith(".pdf")), None)
        if not figure_file:
            figure_file = next((Path(path).name for path in item.get("files", []) if str(path).endswith(".png")), None)
        if not figure_file:
            continue
        caption = item.get("caption", stem.replace("_", " "))
        label = "fig:" + stem.replace("_xfig", "").replace("_", "-")
        lines.extend(
            [
                "\\begin{figure}[!t]",
                "\\centering",
                f"\\includegraphics[width=\\linewidth]{{figures/generated/{figure_file}}}",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\end{figure}",
                "",
            ]
        )
    snippet_path = output_dir / "figure_latex_snippets.tex"
    snippet_path.write_text("\n".join(lines), encoding="utf-8")
    return str(snippet_path)


def build_xfig_style_figures(
    output_dir: Path,
    literature_csv: Path | None = None,
    run_dir: Path | None = None,
) -> dict:
    _ensure_dir(output_dir)
    outputs: dict[str, dict] = {}
    skipped: dict[str, str] = {}

    outputs["fig_01_atlas_pipeline_xfig"] = draw_atlas_pipeline(output_dir)
    outputs["fig_02_multibranch_layers_xfig"] = draw_multibranch_layers(output_dir)
    outputs["fig_03_agentic_explainability_xfig"] = draw_agentic_explainability(output_dir)

    if literature_csv and literature_csv.exists():
        outputs["fig_04_related_work_numeric_summary_xfig"] = draw_literature_numeric_summary(literature_csv, output_dir)
    else:
        skipped["fig_04_related_work_numeric_summary_xfig"] = f"Missing literature CSV: {literature_csv}"

    summary_dir = run_dir / "paper_summary" if run_dir else None
    data_plotters = [
        ("fig_05_main_metrics_heatmap_xfig", draw_main_metrics),
        ("fig_06_ablation_delta_xfig", draw_ablation),
        ("fig_07_latency_realtime_xfig", draw_latency),
        ("fig_08_attention_summary_xfig", draw_attention),
    ]
    for stem, plotter in data_plotters:
        if summary_dir and summary_dir.exists():
            item = plotter(summary_dir, output_dir)
            if item:
                outputs[stem] = item
            else:
                skipped[stem] = f"Required result table missing or empty under {summary_dir}"
        else:
            skipped[stem] = "No run_dir/paper_summary artifact folder was provided or found."

    inventory = {
        "output_dir": str(output_dir),
        "literature_csv": str(literature_csv) if literature_csv else None,
        "run_dir": str(run_dir) if run_dir else None,
        "style": {
            "intent": "XFig-like technical schematic with white background, hard-edged boxes, thin arrows, grouped layers, and muted scientific colors.",
            "formats": ["png", "svg", "pdf"],
            "dpi": 320,
        },
        "outputs": outputs,
        "skipped": skipped,
        "latex_snippets": _latex_snippets(outputs, output_dir),
        "guardrail": "Schematic figures show implemented pipeline structure. Data-backed figures are generated only when corresponding artifact tables exist.",
    }
    (output_dir / "figure_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory
