from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from atlas_idnet.config import write_json


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_weighted": "Precision",
    "recall_weighted": "Recall",
    "f1_weighted": "Weighted F1",
    "f1_macro": "Macro F1",
    "matthews_correlation_coefficient": "MCC",
    "cohen_kappa": "Cohen kappa",
    "macro_false_alarm_rate_fpr": "FAR",
    "macro_miss_rate_fnr": "Miss rate",
}


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required paper-summary table: {path}")
    return pd.read_csv(path)


def _round_numeric(table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    rounded = table.copy()
    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(digits)
    return rounded


def _write_table(table: pd.DataFrame, output_dir: Path, stem: str, caption: str, label: str) -> dict:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False)
    latex = table.to_latex(index=False, escape=True, caption=caption, label=label)
    tex_path.write_text(latex, encoding="utf-8")
    return {"csv": str(csv_path), "latex": str(tex_path), "rows": int(len(table)), "columns": list(table.columns)}


def _main_performance(summary_dir: Path) -> pd.DataFrame:
    table = _read_csv(summary_dir / "table_main_atlas_metrics.csv")
    cols = [
        "dataset",
        "model",
        "device",
        "best_epoch",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "f1_macro",
        "matthews_correlation_coefficient",
        "cohen_kappa",
        "macro_false_alarm_rate_fpr",
        "macro_miss_rate_fnr",
    ]
    table = table[[col for col in cols if col in table.columns]].copy()
    table = table.rename(columns={"dataset": "Dataset", "model": "Model", "device": "Device", "best_epoch": "Best epoch", **METRIC_LABELS})
    return _round_numeric(table)


def _ablation(summary_dir: Path) -> pd.DataFrame:
    table = _read_csv(summary_dir / "table_atlas_branch_ablation.csv")
    cols = [
        "dataset",
        "variant",
        "component_removed",
        "active_feature_count",
        "masked_feature_count",
        "f1_weighted",
        "f1_macro",
        "delta_f1_weighted_vs_full_atlas",
        "delta_f1_macro_vs_full_atlas",
        "macro_false_alarm_rate_fpr",
        "method",
        "retrained",
    ]
    table = table[[col for col in cols if col in table.columns]].copy()
    table = table.rename(
        columns={
            "dataset": "Dataset",
            "variant": "ATLAS variant",
            "component_removed": "Component removed",
            "active_feature_count": "Active features",
            "masked_feature_count": "Masked features",
            "f1_weighted": "Weighted F1",
            "f1_macro": "Macro F1",
            "delta_f1_weighted_vs_full_atlas": "Delta weighted F1",
            "delta_f1_macro_vs_full_atlas": "Delta macro F1",
            "macro_false_alarm_rate_fpr": "FAR",
            "method": "Method",
            "retrained": "Retrained",
        }
    )
    return _round_numeric(table)


def _latency(summary_dir: Path) -> pd.DataFrame:
    table = _read_csv(summary_dir / "table_latency_realtime.csv")
    cols = ["dataset", "device", "feature_dim", "samples", "latency_ms_per_sample", "throughput_samples_per_second", "measured_at_utc"]
    table = table[[col for col in cols if col in table.columns]].copy()
    table = table.rename(
        columns={
            "dataset": "Dataset",
            "device": "Device",
            "feature_dim": "Feature dim.",
            "samples": "Samples",
            "latency_ms_per_sample": "Latency (ms/sample)",
            "throughput_samples_per_second": "Throughput (samples/s)",
            "measured_at_utc": "Measured at UTC",
        }
    )
    return _round_numeric(table)


def _top_features(summary_dir: Path, top_n: int) -> pd.DataFrame:
    table = _read_csv(summary_dir / "table_top_atlas_features.csv")
    cols = ["dataset", "feature", "importance_mean", "importance_std", "baseline_macro_f1", "method", "samples", "repeats"]
    table = table[[col for col in cols if col in table.columns]].copy()
    table = table.groupby("dataset", group_keys=False).head(top_n)
    table = table.rename(
        columns={
            "dataset": "Dataset",
            "feature": "Feature",
            "importance_mean": "Mean macro-F1 drop",
            "importance_std": "Std.",
            "baseline_macro_f1": "Baseline macro-F1",
            "method": "Method",
            "samples": "Samples",
            "repeats": "Repeats",
        }
    )
    return _round_numeric(table)


def _attention(summary_dir: Path) -> pd.DataFrame:
    table = _read_csv(summary_dir / "table_attention_summary.csv")
    branch_col = "attention_branch" if "attention_branch" in table.columns else table.columns[1]
    cols = ["dataset", branch_col, "mean", "std", "min", "50%", "max"]
    table = table[[col for col in cols if col in table.columns]].copy()
    table = table.rename(
        columns={
            "dataset": "Dataset",
            branch_col: "Attention branch",
            "mean": "Mean",
            "std": "Std.",
            "min": "Min.",
            "50%": "Median",
            "max": "Max.",
        }
    )
    return _round_numeric(table)


def _literature_template(output_dir: Path) -> dict:
    table = pd.DataFrame(
        [
            {
                "Study": "Fill after live literature verification",
                "Dataset": "CICIoT2023 / NF-UQ-NIDS-v2",
                "Model": "Published IDS method",
                "Accuracy": "",
                "F1-score": "",
                "FAR": "",
                "Year": "",
                "DOI/source": "",
                "Notes": "Do not use until source, split, task, and metric definitions are verified.",
            }
        ]
    )
    return _write_table(
        table,
        output_dir,
        "ieee_table_literature_comparison_template",
        "Template for literature comparison; values must be filled only after source verification.",
        "tab:literature-comparison-template",
    )


def _narrative(main: pd.DataFrame, latency: pd.DataFrame, output_dir: Path) -> str:
    lines = [
        "# IEEE Access Results Narrative Draft",
        "",
        "The following text is generated from verified ATLAS-IDNet artifact tables. Edit the prose before manuscript submission and keep the exact run directory cited in the reproducibility statement.",
        "",
        "ATLAS-IDNet was evaluated on CICIoT2023 and NF-UQ-NIDS-v2 under the multiclass intrusion-detection setting. The reported metrics include accuracy, weighted F1-score, macro F1-score, Matthews correlation coefficient, Cohen's kappa, false-alarm rate, and miss rate.",
    ]
    for _, row in main.iterrows():
        lines.append(
            f"For {row['Dataset']}, ATLAS-IDNet achieved accuracy {row['Accuracy']:.4f}, weighted F1-score {row['Weighted F1']:.4f}, macro F1-score {row['Macro F1']:.4f}, MCC {row['MCC']:.4f}, and FAR {row['FAR']:.4f} on the verified run."
        )
    if not latency.empty:
        latency_bits = [
            f"{row['Dataset']}: {row['Latency (ms/sample)']:.4f} ms/sample and {row['Throughput (samples/s)']:.2f} samples/s on {row['Device']}"
            for _, row in latency.iterrows()
        ]
        lines.append("The CPU latency analysis supports real-time feasibility, with " + "; ".join(latency_bits) + ".")
    lines.append("The branch-masking ablation table should be interpreted relative to the full ATLAS reference row; negative delta values indicate performance degradation after removing or masking a feature branch.")
    lines.append("The explanation tables report ATLAS checkpoint permutation importance and learned branch-attention statistics; no surrogate model is used for the final explainability evidence.")
    path = output_dir / "ieee_results_narrative.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build_ieee_access_tables(run_dir: Path, top_n_features: int = 10) -> dict:
    summary_dir = run_dir / "paper_summary"
    if not summary_dir.exists():
        raise FileNotFoundError(f"Missing paper_summary folder: {summary_dir}. Run 08_paper_results_summary.ipynb first.")
    output_dir = summary_dir / "ieee_access_tables"
    _reset_output_dir(output_dir)

    main = _main_performance(summary_dir)
    ablation = _ablation(summary_dir)
    latency = _latency(summary_dir)
    top_features = _top_features(summary_dir, top_n_features)
    attention = _attention(summary_dir)

    outputs = {
        "main_performance": _write_table(main, output_dir, "ieee_table_main_performance", "ATLAS-IDNet multiclass intrusion-detection performance.", "tab:main-performance"),
        "ablation": _write_table(ablation, output_dir, "ieee_table_atlas_ablation", "ATLAS-IDNet branch-masking ablation analysis.", "tab:atlas-ablation"),
        "latency": _write_table(latency, output_dir, "ieee_table_realtime_latency", "ATLAS-IDNet real-time inference latency and throughput.", "tab:realtime-latency"),
        "top_features": _write_table(top_features, output_dir, "ieee_table_explainability_features", "Top ATLAS-IDNet permutation-importance features.", "tab:explainability-features"),
        "attention": _write_table(attention, output_dir, "ieee_table_attention_summary", "ATLAS-IDNet branch-attention summary.", "tab:attention-summary"),
        "literature_template": _literature_template(output_dir),
        "narrative": _narrative(main, latency, output_dir),
    }
    inventory = {
        "run_dir": str(run_dir),
        "paper_summary_dir": str(summary_dir),
        "output_dir": str(output_dir),
        "top_n_features": int(top_n_features),
        "outputs": outputs,
        "guardrail": "These IEEE Access tables format existing verified artifacts only. Literature comparison values must be verified separately before use.",
    }
    write_json(output_dir / "ieee_access_table_inventory.json", inventory)
    return inventory

