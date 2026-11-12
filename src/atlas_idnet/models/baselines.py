from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_baselines(seed: int = 42) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
    }


def train_baselines(x_train, y_train, output_dir: Path, seed: int = 42) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = build_baselines(seed)
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump(model, output_dir / f"{name}.joblib")
        fitted[name] = model
    return fitted

