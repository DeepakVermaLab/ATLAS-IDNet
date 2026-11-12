from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_idnet.config import read_json, write_json


def _read_json_if_exists(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _best_metric_row(table: pd.DataFrame, metric: str = "f1_macro") -> dict:
    if table.empty or metric not in table.columns:
        return {}
    idx = table[metric].astype(float).idxmax()
    return table.loc[idx].to_dict()


def _largest_ablation_drop(table: pd.DataFrame, metric: str = "delta_f1_macro_vs_full_atlas") -> dict:
    if table.empty or metric not in table.columns:
        return {}
    candidates = table[table["variant"] != "full_atlas"].copy()
    if candidates.empty:
        return {}
    idx = candidates[metric].astype(float).idxmin()
    return candidates.loc[idx].to_dict()


def build_agentic_report(dataset_name: str, task: str, run_dir: Path, output_dir: Path | None = None) -> dict:
    output_dir = output_dir or (run_dir / dataset_name / task / "agentic")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_dir / dataset_name
    task_dir = dataset_dir / task

    eda = _read_json_if_exists(dataset_dir / "eda" / f"{dataset_name}_eda_summary.json")
    prepared = _read_json_if_exists(task_dir / "preprocessed" / f"{dataset_name}_{task}_prepared_metadata.json")
    atlas_metrics = _read_json_if_exists(task_dir / "models" / "atlas" / "test_metrics.json")
    ablation = _read_csv_if_exists(task_dir / "ablation" / "ablation_results.csv")
    latency = _read_json_if_exists(task_dir / "latency" / "latency.json")
    atlas_importance = _read_csv_if_exists(task_dir / "explainability" / "atlas_permutation_importance.csv")
    attention = _read_csv_if_exists(task_dir / "explainability" / "atlas_attention_summary.csv")

    ablation_reference = ablation.loc[ablation["variant"] == "full_atlas"].iloc[0].to_dict() if not ablation.empty and "variant" in ablation.columns and (ablation["variant"] == "full_atlas").any() else _best_metric_row(ablation)
    ablation_largest_drop = _largest_ablation_drop(ablation)
    top_features = atlas_importance.head(10).to_dict(orient="records") if not atlas_importance.empty else []

    recommendations = []
    if ablation_largest_drop:
        recommendations.append(f"The largest ATLAS branch-masking degradation is {ablation_largest_drop.get('variant')} under macro-F1; use this to justify the corresponding feature branch.")
    if latency:
        recommendations.append("Latency was measured on the runtime used for this run; report device and batch settings with the timing table.")
    if top_features:
        recommendations.append("Use ATLAS permutation importance and branch-attention summaries as the first explainability evidence; add per-class examples before manuscript submission.")

    payload = {
        "dataset": dataset_name,
        "task": task,
        "eda": eda,
        "prepared": prepared,
        "atlas_metrics": atlas_metrics,
        "ablation_reference": ablation_reference,
        "largest_ablation_drop": ablation_largest_drop,
        "latency": latency,
        "top_features": top_features,
        "explainability_method": "ATLAS-IDNet checkpoint permutation importance plus branch attention",
        "surrogate_models_used": False,
        "recommendations": recommendations,
    }
    write_json(output_dir / "agentic_explanation.json", payload)

    lines = [
        f"# ATLAS-IDNet Agentic Explanation Report: {dataset_name} ({task})",
        "",
        "## Run Scope",
        f"- Dataset: {dataset_name}",
        f"- Task: {task}",
        f"- Sample rows: {prepared.get('rows', eda.get('sample_rows', 'not available'))}",
        f"- Feature count: {prepared.get('features', 'not available')}",
        f"- Label column: {prepared.get('label_column', eda.get('label_column', 'not available'))}",
        "",
        "## Detection Result",
    ]
    if atlas_metrics:
        lines.extend(
            [
                f"- ATLAS-IDNet accuracy: {atlas_metrics.get('accuracy', 'not available')}",
                f"- ATLAS-IDNet balanced accuracy: {atlas_metrics.get('balanced_accuracy', 'not available')}",
                f"- ATLAS-IDNet macro-F1: {atlas_metrics.get('f1_macro', 'not available')}",
                f"- ATLAS-IDNet weighted-F1: {atlas_metrics.get('f1_weighted', 'not available')}",
            ]
        )
    else:
        lines.append("- ATLAS-IDNet metrics are not available yet. Run the training notebook first.")
    if ablation_reference:
        lines.append(f"- Full ATLAS ablation reference macro-F1: {ablation_reference.get('f1_macro')}")
    if ablation_largest_drop:
        lines.append(f"- Largest ablation degradation: {ablation_largest_drop.get('variant')} ({ablation_largest_drop.get('delta_f1_macro_vs_full_atlas')})")

    lines.extend(["", "## Explainability Evidence"])
    if top_features:
        for item in top_features[:8]:
            lines.append(f"- {item.get('feature')}: {item.get('importance_mean')}")
    else:
        lines.append("- ATLAS permutation-importance artifacts are not available yet.")
    if not attention.empty:
        lines.append("- Branch-attention summary is available in `atlas_attention_summary.csv` and `atlas_attention_summary.png`.")
    lines.append("- No surrogate model is used for the agentic explanation payload.")

    lines.extend(["", "## Real-Time Feasibility"])
    if latency:
        lines.extend(
            [
                f"- Samples timed: {latency.get('samples')}",
                f"- Latency per sample, ms: {latency.get('latency_ms_per_sample')}",
                f"- Throughput, samples/s: {latency.get('throughput_samples_per_second')}",
            ]
        )
    else:
        lines.append("- Latency artifacts are not available yet. Run the latency stage after training.")

    lines.extend(["", "## Agent Recommendations"])
    lines.extend([f"- {item}" for item in recommendations] or ["- Run EDA, preprocessing, training, ablation, explainability, and latency stages before using this report in the paper."])
    lines.extend(
        [
            "",
            "## Manuscript Guardrails",
            "- Treat all metrics as run-specific until the full dataset split and external dataset validation are complete.",
            "- Report dataset names, split strategy, sample size, runtime device, and output artifact paths with every table.",
            "- Do not claim Q1 acceptance, real-world deployment, or clinical/security production readiness from this benchmark-only evidence.",
        ]
    )
    (output_dir / "agentic_explanation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
