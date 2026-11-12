from __future__ import annotations

import time
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from atlas_idnet.config import write_json
from atlas_idnet.data.preprocessing import PreparedData
from atlas_idnet.models.baselines import train_baselines
from atlas_idnet.models.multimodal import build_model, parameter_count, predict_proba, require_torch
from atlas_idnet.training.metrics import compute_metrics, write_metric_artifacts, write_prediction_artifacts
from sklearn.metrics import f1_score


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train.astype("int64"), minlength=num_classes).astype("float64")
    total = counts.sum()
    weights = np.zeros(num_classes, dtype="float32")
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    if weights[present].size:
        weights[present] = np.minimum(weights[present], np.percentile(weights[present], 95))
        weights[present] = weights[present] / weights[present].mean()
    return weights


def load_torch_checkpoint(torch_module, path: Path, map_location):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


def train_atlas_model(
    prepared: PreparedData,
    output_dir: Path,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    patience: int = 4,
    seed: int = 42,
) -> dict:
    _reset_output_dir(output_dir)
    torch, nn, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(prepared.x_train.shape[1], len(prepared.class_names), prepared.feature_names, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.05)
    criterion = nn.CrossEntropyLoss()

    x_train = torch.tensor(prepared.x_train, dtype=torch.float32)
    y_train = torch.tensor(prepared.y_train, dtype=torch.long)
    x_val = torch.tensor(prepared.x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(prepared.y_val, dtype=torch.long).to(device)

    history = []
    best_score = float("-inf")
    best_epoch = 0
    best_path = output_dir / "atlas_idnet.pt"
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train))
        losses = []
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).detach().cpu())
            val_pred = model(x_val).argmax(dim=1).detach().cpu().numpy()
            val_macro_f1 = float(f1_score(prepared.y_val, val_pred, average="macro", zero_division=0))
            val_weighted_f1 = float(f1_score(prepared.y_val, val_pred, average="weighted", zero_division=0))
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_f1_macro": val_macro_f1,
                "val_f1_weighted": val_weighted_f1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_weighted_f1 > best_score:
            best_score = val_weighted_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": prepared.feature_names,
                    "class_names": prepared.class_names,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "selection_metric": "val_f1_weighted",
                },
                best_path,
            )
        if epoch - best_epoch >= patience:
            break

    checkpoint = load_torch_checkpoint(torch, best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    proba = predict_proba(model.cpu(), prepared.x_test)
    pred = proba.argmax(axis=1)
    metrics = compute_metrics(prepared.y_test, pred, proba)
    metrics.update(
        {
            "model": "ATLAS-IDNet",
            "device": str(device),
            "parameters": parameter_count(model),
            "best_epoch": best_epoch,
            "best_val_f1_weighted": best_score,
            "training_seconds": time.perf_counter() - start_time,
            "loss": "cross_entropy",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "architecture": "multi_branch_residual_attention",
        }
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    write_json(output_dir / "test_metrics.json", metrics)
    write_metric_artifacts(prepared.y_test, pred, prepared.class_names, output_dir)
    write_prediction_artifacts(prepared.y_test, pred, proba, prepared.class_names, output_dir, prefix="test")
    return {"model": model, "metrics": metrics, "probabilities": proba, "predictions": pred}


def train_and_evaluate_baselines(prepared: PreparedData, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = train_baselines(prepared.x_train, prepared.y_train, output_dir, seed=seed)
    results = {}
    rows = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(prepared.x_test)
        else:
            proba = None
        pred = model.predict(prepared.x_test)
        metrics = compute_metrics(prepared.y_test, pred, proba)
        metrics["model"] = name
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "test_metrics.json", metrics)
        write_metric_artifacts(prepared.y_test, pred, prepared.class_names, model_dir)
        write_prediction_artifacts(prepared.y_test, pred, proba, prepared.class_names, model_dir, prefix="test")
        results[name] = metrics
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(output_dir / "baseline_metrics.csv", index=False)
    write_json(output_dir / "baseline_metrics.json", {"models": results})
    return results
