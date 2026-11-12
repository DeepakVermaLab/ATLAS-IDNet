from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from atlas_idnet.data.preprocessing import PreparedData
from atlas_idnet.models.multimodal import infer_feature_groups, predict_proba
from atlas_idnet.training.metrics import compute_metrics


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _masked_features(x: np.ndarray, active_idx: list[int]) -> np.ndarray:
    masked = np.zeros_like(x)
    masked[:, active_idx] = x[:, active_idx]
    return masked


def run_ablation(prepared: PreparedData, output_dir: Path, model) -> pd.DataFrame:
    _reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = infer_feature_groups(prepared.feature_names)
    all_idx = list(range(len(prepared.feature_names)))

    def without(*removed_groups: list[int]) -> list[int]:
        removed = set()
        for group in removed_groups:
            removed.update(group)
        return [idx for idx in all_idx if idx not in removed]

    variants = {
        "full_atlas": all_idx,
        "without_traffic_branch": without(groups.traffic_idx),
        "without_protocol_branch": without(groups.protocol_idx),
        "without_temporal_branch": without(groups.temporal_idx),
        "traffic_branch_only": groups.traffic_idx,
        "traffic_temporal_branches": sorted(set(groups.traffic_idx + groups.temporal_idx)),
    }
    rows = []
    model = model.cpu()
    for name, idx in variants.items():
        x_test = prepared.x_test if name == "full_atlas" else _masked_features(prepared.x_test, idx)
        proba = predict_proba(model, x_test)
        pred = proba.argmax(axis=1)
        metrics = compute_metrics(prepared.y_test, pred, proba)
        metrics.update(
            {
                "variant": name,
                "reference_variant": "full_atlas",
                "active_feature_count": len(idx),
                "masked_feature_count": int(len(prepared.feature_names) - len(idx)),
                "method": "atlas_checkpoint_branch_masking",
                "retrained": False,
                "model": "ATLAS-IDNet",
            }
        )
        rows.append(metrics)

    table = pd.DataFrame(rows)
    reference = table.loc[table["variant"] == "full_atlas"].iloc[0]
    for metric in ["accuracy", "f1_weighted", "f1_macro", "matthews_correlation_coefficient", "macro_false_alarm_rate_fpr"]:
        table[f"delta_{metric}_vs_full_atlas"] = table[metric].astype(float) - float(reference[metric])
    table["component_removed"] = table["variant"].map(
        {
            "full_atlas": "none",
            "without_traffic_branch": "traffic",
            "without_protocol_branch": "protocol",
            "without_temporal_branch": "temporal",
            "traffic_branch_only": "protocol+temporal",
            "traffic_temporal_branches": "protocol",
        }
    )
    table["variant_order"] = table["variant"].map({name: order for order, name in enumerate(variants)})
    table = table.sort_values("variant_order").drop(columns=["variant_order"])
    table.to_csv(output_dir / "ablation_results.csv", index=False)
    table.to_csv(output_dir / "atlas_branch_ablation_results.csv", index=False)
    return table
