from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from atlas_idnet.config import read_json, write_json


MAIN_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "f1_macro",
    "matthews_correlation_coefficient",
    "cohen_kappa",
    "macro_false_alarm_rate_fpr",
    "macro_miss_rate_fnr",
]


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _dataset_dirs(run_dir: Path, task: str) -> list[Path]:
    return sorted(path for path in run_dir.iterdir() if (path / task).is_dir())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _plot_bar(table: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    if table.empty or x not in table.columns or y not in table.columns:
        return
    plt.figure(figsize=(8, 4.8))
    plt.bar(table[x].astype(str), table[y].astype(float), color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"][: len(table)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_grouped_metrics(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    metrics = ["accuracy", "f1_weighted", "f1_macro", "matthews_correlation_coefficient"]
    available = [metric for metric in metrics if metric in table.columns]
    if not available:
        return
    plot = table.set_index("dataset")[available].astype(float)
    ax = plot.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("ATLAS-IDNet main detection performance")
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_ablation_delta(table: pd.DataFrame, path: Path) -> None:
    if table.empty or "delta_f1_weighted_vs_full_atlas" not in table.columns:
        return
    plot = table[table["variant"] != "full_atlas"].copy()
    if plot.empty:
        return
    plot["label"] = plot["dataset"].astype(str) + " | " + plot["variant"].astype(str)
    plot = plot.sort_values("delta_f1_weighted_vs_full_atlas")
    plt.figure(figsize=(11, max(5, 0.42 * len(plot))))
    plt.barh(plot["label"], plot["delta_f1_weighted_vs_full_atlas"].astype(float), color="#b91c1c")
    plt.axvline(0, color="#111827", linewidth=1)
    plt.title("ATLAS branch-masking ablation: weighted-F1 delta vs full ATLAS")
    plt.xlabel("delta weighted-F1")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _plot_top_features(table: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if table.empty:
        return
    for dataset, subset in table.groupby("dataset"):
        top = subset.head(top_n).copy()
        if top.empty:
            continue
        plt.figure(figsize=(9, max(5, 0.4 * len(top))))
        plt.barh(top["feature"][::-1], top["importance_mean"].astype(float)[::-1], color="#0f766e")
        plt.title(f"{dataset}: ATLAS permutation feature importance")
        plt.xlabel("macro-F1 drop")
        plt.tight_layout()
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(dataset))
        plt.savefig(output_dir / f"figure_top_features_{safe_name}.png", dpi=220)
        plt.close()


def build_paper_summary(run_dir: Path, task: str = "multiclass", top_n_features: int = 10) -> dict:
    output_dir = run_dir / "paper_summary"
    _reset_output_dir(output_dir)

    main_rows = []
    ablation_rows = []
    latency_rows = []
    feature_rows = []
    attention_rows = []

    for dataset_dir in _dataset_dirs(run_dir, task):
        dataset = dataset_dir.name
        task_dir = dataset_dir / task

        metrics_path = task_dir / "models" / "atlas" / "test_metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            row = {
                "dataset": dataset,
                "model": "ATLAS-IDNet",
                "device": metrics.get("device"),
                "best_epoch": metrics.get("best_epoch"),
                "parameters": metrics.get("parameters"),
            }
            row.update({metric: metrics.get(metric) for metric in MAIN_METRICS})
            main_rows.append(row)

        ablation = _read_csv(task_dir / "ablation" / "atlas_branch_ablation_results.csv")
        if ablation.empty:
            ablation = _read_csv(task_dir / "ablation" / "ablation_results.csv")
        if not ablation.empty:
            ablation.insert(0, "dataset", dataset)
            ablation_rows.extend(ablation.to_dict(orient="records"))

        latency_path = task_dir / "latency" / "latency.json"
        if latency_path.exists():
            latency = read_json(latency_path)
            latency_rows.append(
                {
                    "dataset": dataset,
                    "device": latency.get("device"),
                    "feature_dim": latency.get("feature_dim"),
                    "samples": latency.get("samples"),
                    "latency_ms_per_sample": latency.get("latency_ms_per_sample"),
                    "throughput_samples_per_second": latency.get("throughput_samples_per_second"),
                    "measured_at_utc": latency.get("measured_at_utc"),
                }
            )

        importance = _read_csv(task_dir / "explainability" / "atlas_permutation_importance.csv")
        if not importance.empty:
            importance = importance.head(top_n_features).copy()
            importance.insert(0, "dataset", dataset)
            feature_rows.extend(importance.to_dict(orient="records"))

        attention = _read_csv(task_dir / "explainability" / "atlas_attention_summary.csv")
        if not attention.empty:
            attention = attention.rename(columns={attention.columns[0]: "attention_branch"})
            attention.insert(0, "dataset", dataset)
            attention_rows.extend(attention.to_dict(orient="records"))

    main_table = pd.DataFrame(main_rows)
    ablation_table = pd.DataFrame(ablation_rows)
    latency_table = pd.DataFrame(latency_rows)
    feature_table = pd.DataFrame(feature_rows)
    attention_table = pd.DataFrame(attention_rows)

    main_table.to_csv(output_dir / "table_main_atlas_metrics.csv", index=False)
    ablation_table.to_csv(output_dir / "table_atlas_branch_ablation.csv", index=False)
    latency_table.to_csv(output_dir / "table_latency_realtime.csv", index=False)
    feature_table.to_csv(output_dir / "table_top_atlas_features.csv", index=False)
    attention_table.to_csv(output_dir / "table_attention_summary.csv", index=False)

    _plot_grouped_metrics(main_table, output_dir / "figure_main_metrics.png")
    _plot_bar(latency_table, "dataset", "latency_ms_per_sample", output_dir / "figure_latency_ms_per_sample.png", "ATLAS-IDNet CPU inference latency", "ms/sample")
    _plot_bar(latency_table, "dataset", "throughput_samples_per_second", output_dir / "figure_throughput_samples_per_second.png", "ATLAS-IDNet CPU inference throughput", "samples/second")
    _plot_ablation_delta(ablation_table, output_dir / "figure_ablation_delta_weighted_f1.png")
    _plot_top_features(feature_table, output_dir, top_n_features)

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "task": task,
        "datasets": sorted(main_table["dataset"].tolist()) if not main_table.empty else [],
        "tables": {
            "main_metrics": str(output_dir / "table_main_atlas_metrics.csv"),
            "ablation": str(output_dir / "table_atlas_branch_ablation.csv"),
            "latency": str(output_dir / "table_latency_realtime.csv"),
            "top_features": str(output_dir / "table_top_atlas_features.csv"),
            "attention": str(output_dir / "table_attention_summary.csv"),
        },
        "figures": sorted(str(path) for path in output_dir.glob("figure_*.png")),
        "paper_guardrail": "These tables summarize already-generated artifacts only; no model training or evaluation is rerun.",
    }
    write_json(output_dir / "paper_results_summary.json", summary)
    return summary

