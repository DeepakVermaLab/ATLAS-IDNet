# ATLAS-IDNet Implementation Runbook

This implementation is organized so notebooks stay thin and all reusable logic lives under `implementation/src/atlas_idnet`.

## Datasets

- CICIoT2023: primary training and internal evaluation dataset.
- NF-UQ-NIDS-v2: external validation/generalization dataset.

Default local paths:

- `/Volumes/LocalDrive1/Dataset/Security/CICIoT2023/raw/wataiData`
- `/Volumes/LocalDrive1/Dataset/Security/NF-UQ-NIDS-v2/raw/NF-UQ-NIDS-v2.csv`

Default Colab/Drive paths:

- `/content/drive/MyDrive/projects/Dataset/Security/CICIoT2023`
- `/content/drive/MyDrive/projects/Dataset/Security/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv`

## Result Location

The target Drive folder ID is `1llQSxhyHiks9lOTX8ZZRMtpn1YnGx8AR`.

When Google Drive is mounted in Colab, outputs are written to:

`/content/drive/MyDrive/projects/ATLAS-IDNet/results/<run_name>/`

When running locally without mounted Drive, outputs are written to:

`implementation/results/<run_name>/`

## Output Tree

Each run produces:

- `config.json`: exact experiment configuration.
- `runtime_provenance.json`: Python, OS, CPU/RAM, package versions, CUDA/GPU details where available.
- `logs/`: stage logs.
- `<dataset>/eda/`: manifest, label counts, missing-value tables, numeric summaries, EDA plots.
- `<dataset>/<task>/preprocessed/`: train/validation/test arrays, scaler, label encoder, metadata.
- `<dataset>/<task>/models/baselines/`: removed during the active ATLAS-only training run if stale baseline artifacts exist.
- `<dataset>/<task>/models/atlas/`: ATLAS-IDNet checkpoint, training history, aggregate metrics, confusion matrix, classification report, per-class detection metrics, and prediction CSV.
- `<dataset>/<task>/ablation/`: ATLAS checkpoint branch-masking ablation results.
- `<dataset>/<task>/latency/`: latency/throughput JSON plus batch-wise p50/p95/p99 latency table.
- `<dataset>/<task>/explainability/`: ATLAS permutation importance and branch-attention tables and plots.
- `<dataset>/<task>/agentic/`: grounded agentic explanation JSON and Markdown report.
- `paper_summary/`: paper-ready metric, ablation, latency, attention, and feature-importance tables plus summary figures.
- `paper_summary/ieee_access_tables/`: IEEE Access formatted CSV/LaTeX tables and a results-narrative draft.

## Paper Metrics Captured

The metric artifacts include:

- accuracy and error rate
- balanced accuracy
- precision, recall, and F1 with macro, micro, and weighted averaging
- ROC-AUC and PR-AUC when probabilities are available
- log loss when probabilities are available
- Matthews correlation coefficient
- Cohen kappa
- confusion matrix
- per-class TP, TN, FP, FN
- detection rate/TPR, specificity/TNR, false alarm rate/FPR, miss rate/FNR
- prediction-level CSV with true labels, predicted labels, and class probabilities

## Recommended Run Order

1. `00_setup_and_environment.ipynb`
2. `01_eda.ipynb`
3. `02_preprocess.ipynb`
4. `03_train_and_evaluate.ipynb`
5. `04_ablation.ipynb`
6. `05_latency.ipynb`
7. `06_explainability_agent.ipynb`
8. `08_paper_results_summary.ipynb`
9. `09_ieee_access_results_tables.ipynb`

For a quick smoke run, set `MAX_ROWS = 20000` in notebooks. For paper-facing results, increase or remove `MAX_ROWS`, keep the same seed, record the runtime device, and inspect all generated metrics.

## Terminal Equivalents

From the repo root:

```bash
python implementation/scripts/00_check_environment.py --run-name smoke_check
python implementation/scripts/01_run_eda.py --dataset CICIoT2023 --max-rows 50000 --run-name ciciot_eda
python implementation/scripts/02_preprocess.py --dataset CICIoT2023 --task multiclass --max-rows 100000 --run-name ciciot_model
python implementation/scripts/03_train_models.py --dataset CICIoT2023 --task multiclass --max-rows 100000 --run-dir implementation/results/ciciot_model --skip-baselines
python implementation/scripts/04_run_ablation.py --dataset CICIoT2023 --task multiclass --run-dir implementation/results/ciciot_model
python implementation/scripts/05_run_latency.py --dataset CICIoT2023 --task multiclass --run-dir implementation/results/ciciot_model
python implementation/scripts/06_explain_agent.py --dataset CICIoT2023 --task multiclass --run-dir implementation/results/ciciot_model
python implementation/scripts/08_paper_results_summary.py --task multiclass --run-dir implementation/results/ciciot_model
python implementation/scripts/09_ieee_access_results_tables.py --task multiclass --run-dir implementation/results/ciciot_model
```

## Manuscript Boundary

The code produces evidence artifacts, but the paper should only report metrics from notebooks or scripts that were actually executed. Keep the generated run directory with the manuscript tables.
