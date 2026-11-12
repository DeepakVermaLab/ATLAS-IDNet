from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def per_class_metrics(y_true, y_pred, class_names: list[str] | None = None) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    rows = []
    for label in labels:
        tp = int(((y_true_arr == label) & (y_pred_arr == label)).sum())
        tn = int(((y_true_arr != label) & (y_pred_arr != label)).sum())
        fp = int(((y_true_arr != label) & (y_pred_arr == label)).sum())
        fn = int(((y_true_arr == label) & (y_pred_arr != label)).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "class_id": int(label),
                "class_name": class_names[label] if class_names and label < len(class_names) else str(label),
                "support": int((y_true_arr == label).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall_detection_rate_tpr": recall,
                "specificity_tnr": specificity,
                "false_alarm_rate_fpr": _safe_div(fp, fp + tn),
                "miss_rate_fnr": _safe_div(fn, fn + tp),
                "negative_predictive_value": _safe_div(tn, tn + fn),
                "false_discovery_rate": _safe_div(fp, fp + tp),
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    per_class = per_class_metrics(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "error_rate": float(1.0 - accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_detection_rate_tpr": float(per_class["recall_detection_rate_tpr"].mean()),
        "macro_specificity_tnr": float(per_class["specificity_tnr"].mean()),
        "macro_false_alarm_rate_fpr": float(per_class["false_alarm_rate_fpr"].mean()),
        "macro_miss_rate_fnr": float(per_class["miss_rate_fnr"].mean()),
    }
    if len(labels) == 2:
        positive_label = int(max(labels))
        positive_row = per_class.loc[per_class["class_id"] == positive_label].iloc[0]
        metrics.update(
            {
                "binary_positive_class_id": positive_label,
                "attack_detection_rate_tpr": float(positive_row["recall_detection_rate_tpr"]),
                "false_alarm_rate_fpr": float(positive_row["false_alarm_rate_fpr"]),
                "miss_rate_fnr": float(positive_row["miss_rate_fnr"]),
                "specificity_tnr": float(positive_row["specificity_tnr"]),
            }
        )
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["pr_auc_average_precision"] = float(average_precision_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
                y_bin = label_binarize(y_true, classes=list(range(y_proba.shape[1])))
                metrics["pr_auc_ovr_macro"] = float(average_precision_score(y_bin, y_proba, average="macro"))
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=list(range(y_proba.shape[1]))))
        except Exception:
            metrics["probability_metric_error"] = "not_computable"
    return metrics


def write_metric_artifacts(y_true, y_pred, class_names: list[str], output_dir) -> None:
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=_target_names(class_names, y_true, y_pred),
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=_target_names(class_names, y_true, y_pred), columns=_target_names(class_names, y_true, y_pred)).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    per_class_metrics(y_true, y_pred, class_names).to_csv(output_dir / "per_class_detection_metrics.csv", index=False)


def write_prediction_artifacts(y_true, y_pred, y_proba, class_names: list[str], output_dir, prefix: str = "test") -> None:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    frame["y_true_label"] = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in y_true]
    frame["y_pred_label"] = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in y_pred]
    if y_proba is not None:
        for idx in range(y_proba.shape[1]):
            name = class_names[idx] if idx < len(class_names) else str(idx)
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            frame[f"proba_{idx}_{safe_name}"] = y_proba[:, idx]
    frame.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)


def _target_names(class_names: list[str], y_true, y_pred) -> list[str]:
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return [class_names[i] if i < len(class_names) else str(i) for i in labels]
