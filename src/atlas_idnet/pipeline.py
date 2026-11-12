from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd

from atlas_idnet.config import ExperimentConfig, default_config, now_stamp, resolve_results_root, save_config, write_json
from atlas_idnet.data.discovery import discover_dataset_files, write_dataset_manifest
from atlas_idnet.data.eda import run_eda
from atlas_idnet.data.preprocessing import PreparedData, load_prepared, prepare_dataset
from atlas_idnet.explainability.agent import build_agentic_report
from atlas_idnet.explainability.feature_importance import run_attention_explanation, run_feature_importance
from atlas_idnet.training.ablation import run_ablation
from atlas_idnet.training.latency import measure_latency
from atlas_idnet.training.trainer import load_torch_checkpoint, train_and_evaluate_baselines, train_atlas_model
from atlas_idnet.utils.logging_utils import get_logger
from atlas_idnet.utils.provenance import write_runtime_provenance


def selected_datasets(cfg: ExperimentConfig, dataset: str = "all"):
    if dataset == "all":
        return cfg.datasets.values()
    if dataset not in cfg.datasets:
        raise KeyError(f"Unknown dataset {dataset}. Available: {list(cfg.datasets)}")
    return [cfg.datasets[dataset]]


def create_run_dir(cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    name = run_name or f"{cfg.project_name}_{now_stamp()}"
    run_dir = resolve_results_root(cfg, name)
    save_config(cfg, run_dir / "config.json")
    write_json(run_dir / "run_metadata.json", {"project": cfg.project_name, "run_name": name, "results_root": str(run_dir)})
    write_runtime_provenance(run_dir)
    return run_dir


def run_environment_check(cfg: ExperimentConfig | None = None, run_dir: Path | None = None) -> dict:
    cfg = cfg or default_config()
    run_dir = run_dir or create_run_dir(cfg, "environment_check")
    logger = get_logger("atlas_idnet.environment", run_dir / "logs" / "environment.log")
    payload = {"datasets": {}, "run_dir": str(run_dir), "runtime_provenance_file": str(run_dir / "runtime_provenance.json")}
    for dataset in cfg.datasets.values():
        files = discover_dataset_files(dataset)
        payload["datasets"][dataset.name] = {
            "file_count": len(files),
            "total_bytes": int(sum(p.stat().st_size for p in files)),
            "preview": [str(p) for p in files[:10]],
        }
        logger.info("%s: %s files", dataset.name, len(files))
    write_json(run_dir / "environment_check.json", payload)
    return payload


def run_eda_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all", max_rows: int | None = None) -> dict:
    logger = get_logger("atlas_idnet.eda", run_dir / "logs" / "eda.log")
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        logger.info("Running EDA for %s", ds.name)
        ds_dir = run_dir / ds.name / "eda"
        outputs[ds.name] = run_eda(ds, ds_dir, max_rows=max_rows)
    write_json(run_dir / "eda_index.json", outputs)
    return outputs


def run_manifest_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all") -> dict:
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        outputs[ds.name] = write_dataset_manifest(ds, run_dir / ds.name / "manifest")
    write_json(run_dir / "manifest_index.json", outputs)
    return outputs


def run_preprocess_stage(
    cfg: ExperimentConfig,
    run_dir: Path,
    dataset: str = "all",
    task: str = "multiclass",
    max_rows: int | None = None,
) -> dict[str, PreparedData]:
    logger = get_logger("atlas_idnet.preprocess", run_dir / "logs" / "preprocess.log")
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        logger.info("Preprocessing %s as %s", ds.name, task)
        output_dir = run_dir / ds.name / task / "preprocessed"
        outputs[ds.name] = prepare_dataset(
            ds,
            output_dir,
            task=task,
            max_rows=max_rows,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
    return outputs


def prepared_path(run_dir: Path, dataset_name: str, task: str) -> Path:
    return run_dir / dataset_name / task / "preprocessed" / f"{dataset_name}_{task}_prepared.npz"


def load_or_prepare(cfg: ExperimentConfig, run_dir: Path, dataset_name: str, task: str, max_rows: int | None = None) -> PreparedData:
    path = prepared_path(run_dir, dataset_name, task)
    if path.exists():
        return load_prepared(path)
    ds = cfg.datasets[dataset_name]
    return prepare_dataset(
        ds,
        path.parent,
        task=task,
        max_rows=max_rows,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
    )


def run_training_stage(
    cfg: ExperimentConfig,
    run_dir: Path,
    dataset: str = "all",
    task: str = "multiclass",
    max_rows: int | None = None,
    train_baselines: bool = True,
    train_atlas: bool = True,
) -> dict:
    logger = get_logger("atlas_idnet.training", run_dir / "logs" / "training.log")
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        logger.info("Training models for %s/%s", ds.name, task)
        prepared = load_or_prepare(cfg, run_dir, ds.name, task, max_rows=max_rows)
        ds_outputs = {}
        if train_baselines:
            ds_outputs["baselines"] = train_and_evaluate_baselines(prepared, run_dir / ds.name / task / "models" / "baselines", seed=cfg.seed)
        else:
            stale_baseline_dir = run_dir / ds.name / task / "models" / "baselines"
            if stale_baseline_dir.exists():
                shutil.rmtree(stale_baseline_dir)
            ds_outputs["baselines_removed"] = str(stale_baseline_dir)
        if train_atlas:
            ds_outputs["atlas"] = train_atlas_model(
                prepared,
                run_dir / ds.name / task / "models" / "atlas",
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                patience=cfg.patience,
                seed=cfg.seed,
            )["metrics"]
        outputs[ds.name] = ds_outputs
    write_json(run_dir / f"training_index_{task}.json", outputs)
    return outputs


def run_ablation_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all", task: str = "multiclass", max_rows: int | None = None) -> dict:
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        prepared = load_or_prepare(cfg, run_dir, ds.name, task, max_rows=max_rows)
        atlas_checkpoint = run_dir / ds.name / task / "models" / "atlas" / "atlas_idnet.pt"
        if not atlas_checkpoint.exists():
            outputs[ds.name] = [{"error": f"Missing ATLAS checkpoint: {atlas_checkpoint}"}]
            continue
        from atlas_idnet.models.multimodal import build_model, require_torch

        torch, _, _ = require_torch()
        model = build_model(prepared.x_train.shape[1], len(prepared.class_names), prepared.feature_names, cfg.hidden_dim, cfg.dropout)
        checkpoint = load_torch_checkpoint(torch, atlas_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        table = run_ablation(prepared, run_dir / ds.name / task / "ablation", model)
        outputs[ds.name] = table.to_dict(orient="records")
    write_json(run_dir / f"ablation_index_{task}.json", outputs)
    return outputs


def run_explainability_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all", task: str = "multiclass", max_rows: int | None = None) -> dict:
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        prepared = load_or_prepare(cfg, run_dir, ds.name, task, max_rows=max_rows)
        explain_dir = run_dir / ds.name / task / "explainability"
        atlas_checkpoint = run_dir / ds.name / task / "models" / "atlas" / "atlas_idnet.pt"
        outputs[ds.name] = {}
        if atlas_checkpoint.exists():
            from atlas_idnet.models.multimodal import build_model, require_torch

            torch, _, _ = require_torch()
            model = build_model(prepared.x_train.shape[1], len(prepared.class_names), prepared.feature_names, cfg.hidden_dim, cfg.dropout)
            checkpoint = load_torch_checkpoint(torch, atlas_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint["model_state"])
            fi = run_feature_importance(prepared, model, explain_dir, seed=cfg.seed, permutation_samples=cfg.explain_samples)
            outputs[ds.name]["atlas_permutation_rows"] = int(len(fi["atlas_permutation"]))
            outputs[ds.name]["feature_group_rows"] = int(len(fi["groups"]))
            run_attention_explanation(model, prepared, explain_dir, samples=cfg.explain_samples)
            outputs[ds.name]["attention"] = "written"
            outputs[ds.name]["surrogate_models_used"] = False
        else:
            outputs[ds.name]["error"] = f"Missing ATLAS checkpoint: {atlas_checkpoint}"
        build_agentic_report(ds.name, task, run_dir, run_dir / ds.name / task / "agentic")
    write_json(run_dir / f"explainability_index_{task}.json", outputs)
    return outputs


def run_latency_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all", task: str = "multiclass", max_rows: int | None = None) -> dict:
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        prepared = load_or_prepare(cfg, run_dir, ds.name, task, max_rows=max_rows)
        atlas_checkpoint = run_dir / ds.name / task / "models" / "atlas" / "atlas_idnet.pt"
        if not atlas_checkpoint.exists():
            outputs[ds.name] = {"error": f"Missing ATLAS checkpoint: {atlas_checkpoint}"}
            continue
        from atlas_idnet.models.multimodal import build_model, require_torch

        torch, _, _ = require_torch()
        model = build_model(prepared.x_train.shape[1], len(prepared.class_names), prepared.feature_names, cfg.hidden_dim, cfg.dropout)
        checkpoint = load_torch_checkpoint(torch, atlas_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        outputs[ds.name] = measure_latency(model, prepared.x_test, run_dir / ds.name / task / "latency", samples=cfg.latency_samples)
    write_json(run_dir / f"latency_index_{task}.json", outputs)
    return outputs


def run_agentic_stage(cfg: ExperimentConfig, run_dir: Path, dataset: str = "all", task: str = "multiclass") -> dict:
    outputs = {}
    for ds in selected_datasets(cfg, dataset):
        outputs[ds.name] = build_agentic_report(ds.name, task, run_dir, run_dir / ds.name / task / "agentic")
    write_json(run_dir / f"agentic_index_{task}.json", outputs)
    return outputs


def build_summary_tables(run_dir: Path, task: str = "multiclass") -> None:
    rows = []
    for metrics_path in run_dir.glob(f"*/{task}/models/atlas/test_metrics.json"):
        dataset_name = metrics_path.parents[3].name
        payload = pd.Series(__import__("json").loads(metrics_path.read_text(encoding="utf-8"))).to_dict()
        payload["dataset"] = dataset_name
        rows.append(payload)
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / f"summary_atlas_metrics_{task}.csv", index=False)


def run_full_pipeline(
    cfg: ExperimentConfig | None = None,
    dataset: str = "all",
    task: str = "multiclass",
    run_name: str | None = None,
    max_rows: int | None = None,
) -> Path:
    cfg = cfg or default_config()
    run_dir = create_run_dir(cfg, run_name)
    run_environment_check(cfg, run_dir)
    run_eda_stage(cfg, run_dir, dataset=dataset, max_rows=max_rows)
    run_preprocess_stage(cfg, run_dir, dataset=dataset, task=task, max_rows=max_rows)
    run_training_stage(cfg, run_dir, dataset=dataset, task=task, max_rows=max_rows, train_baselines=False, train_atlas=True)
    run_ablation_stage(cfg, run_dir, dataset=dataset, task=task, max_rows=max_rows)
    run_latency_stage(cfg, run_dir, dataset=dataset, task=task, max_rows=max_rows)
    run_explainability_stage(cfg, run_dir, dataset=dataset, task=task, max_rows=max_rows)
    run_agentic_stage(cfg, run_dir, dataset=dataset, task=task)
    build_summary_tables(run_dir, task=task)
    return run_dir
