from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from atlas_idnet.config import write_json
from atlas_idnet.data.preprocessing import PreparedData
from atlas_idnet.models.multimodal import infer_feature_groups, require_torch


def _plot_importance(table: pd.DataFrame, value_col: str, output_path: Path, title: str, top_k: int) -> None:
    top = table.head(top_k).copy()
    if top.empty:
        return
    plt.figure(figsize=(11, max(5, min(12, top_k * 0.32))))
    top.sort_values(value_col).plot(x="feature", y=value_col, kind="barh", legend=False)
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _reset_explainability_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _predict_labels(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    model.eval()
    labels = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            labels.append(model(batch).argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(labels)


def run_feature_importance(
    prepared: PreparedData,
    model,
    output_dir: Path,
    seed: int = 42,
    top_k: int = 30,
    permutation_samples: int = 4000,
    repeats: int = 3,
) -> dict:
    _reset_explainability_dir(output_dir)
    torch, _, _ = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    rng = np.random.default_rng(seed)

    x_perm = prepared.x_test[: min(permutation_samples, len(prepared.x_test))]
    y_perm = prepared.y_test[: min(permutation_samples, len(prepared.y_test))]
    if len(x_perm) > 0 and len(np.unique(y_perm)) > 1:
        baseline_pred = _predict_labels(model, x_perm)
        baseline_macro_f1 = float(f1_score(y_perm, baseline_pred, average="macro", zero_division=0))
        rows = []
        for feature_index, feature_name in enumerate(prepared.feature_names):
            drops = []
            for _ in range(repeats):
                x_shuffled = x_perm.copy()
                x_shuffled[:, feature_index] = rng.permutation(x_shuffled[:, feature_index])
                pred = _predict_labels(model, x_shuffled)
                shuffled_score = float(f1_score(y_perm, pred, average="macro", zero_division=0))
                drops.append(baseline_macro_f1 - shuffled_score)
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature": feature_name,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops)),
                    "baseline_macro_f1": baseline_macro_f1,
                    "method": "atlas_checkpoint_permutation_macro_f1_drop",
                    "samples": int(len(x_perm)),
                    "repeats": int(repeats),
                }
            )
        permutation = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
        permutation.to_csv(output_dir / "atlas_permutation_importance.csv", index=False)
        _plot_importance(
            permutation,
            "importance_mean",
            output_dir / "atlas_permutation_importance.png",
            "ATLAS-IDNet permutation importance",
            top_k,
        )
    else:
        permutation = pd.DataFrame(columns=["feature_index", "feature", "importance_mean", "importance_std"])

    groups = infer_feature_groups(prepared.feature_names)
    by_index = permutation.set_index("feature_index") if not permutation.empty else pd.DataFrame()

    def _group_importance(indexes: list[int]) -> float:
        if by_index.empty:
            return 0.0
        available = [idx for idx in indexes if idx in by_index.index]
        if not available:
            return 0.0
        return float(by_index.loc[available]["importance_mean"].sum())

    group_rows = [
        {
            "group": "traffic",
            "feature_count": len(groups.traffic_idx),
            "importance_sum": _group_importance(groups.traffic_idx),
        },
        {
            "group": "protocol",
            "feature_count": len(groups.protocol_idx),
            "importance_sum": _group_importance(groups.protocol_idx),
        },
        {
            "group": "temporal",
            "feature_count": len(groups.temporal_idx),
            "importance_sum": _group_importance(groups.temporal_idx),
        },
    ]
    groups_table = pd.DataFrame(group_rows).sort_values("importance_sum", ascending=False)
    groups_table.to_csv(output_dir / "atlas_feature_group_importance.csv", index=False)
    write_json(
        output_dir / "feature_importance_summary.json",
        {
            "method": "ATLAS-IDNet checkpoint permutation importance and branch attention",
            "surrogate_models_used": False,
            "top_atlas_permutation_features": permutation.head(15).to_dict(orient="records"),
            "feature_groups": groups_table.to_dict(orient="records"),
        },
    )
    return {
        "atlas_permutation": permutation,
        "groups": groups_table,
    }


def run_attention_explanation(model, prepared: PreparedData, output_dir: Path, samples: int = 4096) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, _, _ = require_torch()
    x = prepared.x_test[: min(samples, len(prepared.x_test))]
    if len(x) == 0:
        raise ValueError("No test samples available for attention explanation")
    model.eval()
    weights = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            batch = torch.tensor(x[start : start + 4096], dtype=torch.float32)
            _, batch_weights = model(batch, return_attention=True)
            weights.append(batch_weights.cpu().numpy())
    arr = np.vstack(weights)
    table = pd.DataFrame(arr, columns=["traffic_attention", "protocol_attention", "temporal_attention"])
    summary = table.describe().T
    table.to_csv(output_dir / "atlas_attention_weights.csv", index=False)
    summary.to_csv(output_dir / "atlas_attention_summary.csv")

    plt.figure(figsize=(8, 5))
    summary["mean"].plot(kind="bar")
    plt.title("ATLAS-IDNet mean branch attention")
    plt.ylabel("attention weight")
    plt.tight_layout()
    plt.savefig(output_dir / "atlas_attention_summary.png", dpi=180)
    plt.close()
    return table
