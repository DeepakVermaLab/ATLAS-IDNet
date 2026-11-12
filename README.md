# ATLAS-IDNet

**A reproducible, end-to-end research implementation for lightweight, evidence-grounded network-intrusion analysis.**

ATLAS-IDNet combines a complete exploratory-data-analysis and preprocessing pipeline with a controlled multi-branch detector evaluation, strong classical baselines, capacity-matched neural controls, calibration, rare-class analysis, CPU profiling, explanation diagnostics, and deterministic artifact reporting.

Public recorded-replay companion: <https://deepakvermalab.github.io/ATLAS-IDNet-Simulation/>

> **Repository status:** Private research repository. Raw datasets are not committed. The retained notebooks, logs, manifests, and evidence files are real project artifacts; synthetic smoke tests are identified as code-validation tests and are not presented as research results.

## What is included

| Area | Location | Purpose |
|---|---|---|
| Dataset acquisition | `download/` | Reproducible acquisition notebooks and helpers for CICIoT2023 and NF-UQ-NIDS-v2 |
| Drive transfer | `upload/` | Dataset transfer helpers and operational documentation |
| End-to-end pipeline | `notebooks/`, `src/`, `scripts/` | Environment setup, EDA, preprocessing, training, ablation, latency, explainability, reporting, and paper-facing summaries |
| Controlled protocol | `v2/` | Five-seed publication protocol, matched baselines, diagnostics, evidence agent, tests, and deployment demonstrator |
| Verified evidence | `evidence/v2/` | Aggregate ledgers, paired tests, preparation metadata, seed-42 diagnostic records, and final implementation audit |
| Printed execution evidence | Executed notebooks and `results/` | Preserved notebook outputs and environment/log artifacts |

## System view

```text
Raw CICIoT2023 / NF-UQ-NIDS-v2 records
                 |
                 v
     discovery -> EDA -> cleaning -> split audit
                 |
                 v
   train-only imputation, clipping, and scaling
                 |
                 v
  semantic feature branches + residual global context
                 |
                 v
       attention fusion + gated classifier
                 |
                 v
 repeated evaluation, controls, calibration, latency,
 faithfulness, stability, and deterministic reporting
```

The `v2` detector assigns features to protocol/transport, time-derived tabular, and remaining traffic/numeric branches. Branch representations are fused by learned attention and mixed with a residual global-context pathway. The time-derived branch processes tabular descriptors; it is **not** sequence modeling.

## Recommended execution order

### A. Complete end-to-end pipeline

1. `download/notebooks/01_download_ciciot2023.ipynb`
2. `download/notebooks/02_download_nf_uq_nids_v2.ipynb`
3. `upload/notebooks/01_upload_ciciot2023_to_drive.ipynb`
4. `upload/notebooks/02_upload_nf_uq_nids_v2_to_drive.ipynb`
5. `notebooks/00_setup_and_environment.ipynb`
6. `notebooks/01_eda.ipynb`
7. `notebooks/02_preprocess.ipynb`
8. `notebooks/03_train_and_evaluate.ipynb`
9. `notebooks/04_ablation.ipynb`
10. `notebooks/05_latency.ipynb`
11. `notebooks/06_explainability_agent.ipynb`
12. `notebooks/08_paper_results_summary.ipynb`
13. `notebooks/09_ieee_access_results_tables.ipynb`

### B. Controlled v2 publication protocol

1. `v2/notebooks/00_setup_and_paths.ipynb`
2. `v2/notebooks/01_prepare_and_cpu_pilot.ipynb`
3. `v2/notebooks/04_prepare_publication_protocols.ipynb`
4. `v2/notebooks/05_publication_controlled_experiments.ipynb`
5. `v2/notebooks/06_diagnostics_and_evidence_agent.ipynb`

The compact pilot notebooks `v2/notebooks/02_controlled_experiments.ipynb` and `03_latency_explanations_summary.ipynb` are retained as development-stage protocol checks, not as the final publication evidence.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r v2/requirements.txt
```

The verified v2 environment used Python 3.11 with PyTorch, NumPy, pandas, SciPy, scikit-learn, joblib, matplotlib, and nbformat. Dataset locations are configuration inputs; no machine-specific path is required by the model package itself.

## Command-line pipeline

The original pipeline can be run from this repository root:

```bash
export PYTHONPATH="$PWD/src"
python scripts/00_check_environment.py
python scripts/01_run_eda.py
python scripts/02_preprocess.py
python scripts/03_train_models.py
python scripts/04_run_ablation.py
python scripts/05_run_latency.py
python scripts/06_explain_agent.py
```

The v2 tests use the isolated package:

```bash
export PYTHONPATH="$PWD/v2/src"
python v2/tests/test_sampling.py
python v2/tests/test_rare_split.py
python v2/tests/smoke.py
```

## Verified experimental scope

The completed v2 matrix contains 180 immutable runs across two datasets and two protocols:

- natural-prior evaluation without class balancing;
- a separately reported class-capped minority challenge;
- seeds 42--46;
- nine neural variants and four classical baselines where specified;
- parameter-count controls, calibration, per-class metrics, paired tests, CPU latency and memory profiling, and explanation diagnostics.

The complete protocol summary and measured artifact inventory are recorded in `v2/EXECUTION_STATUS.md` and `evidence/v2/`. The final five-seed ATLAS means retained in the audit are:

| Dataset | Protocol | Accuracy | Weighted F1 | Macro F1 |
|---|---|---:|---:|---:|
| CICIoT2023 | Natural prior | 0.9775 | 0.9806 | 0.6496 |
| CICIoT2023 | Minority challenge | 0.7325 | 0.7339 | 0.7087 |
| NF-UQ-NIDS-v2 | Natural prior | 0.8135 | 0.8624 | 0.6139 |
| NF-UQ-NIDS-v2 | Minority challenge | 0.9013 | 0.9004 | 0.8885 |

These values are not universal superiority claims. Random Forest or Extra Trees is stronger in each matched predictive comparison and that negative result is intentionally preserved.

## Evidence and logs

- Executed notebook outputs are retained in the `.ipynb` files.
- `results/scaffold_environment_check/logs/environment.log` records the original environment check.
- `evidence/v2/*/all_runs.csv` contains the retained run ledgers.
- `evidence/v2/*/seed_summary.csv` and `paired_tests.csv` contain repeated-seed summaries and paired analyses.
- `evidence/v2/final_implementation_audit.csv` is the four-protocol implementation audit.
- `v2/deployment_demo/checksums.sha256` verifies the private replay bundles.

See [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) for the file-level map and [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for operational details.

## Scientific boundaries

- Fixed stratified splits do not establish temporal, cross-site, or cross-device generalization.
- Natural-prior and class-capped challenge results are not interchangeable.
- Attention and occlusion outputs are diagnostics, not causal explanations.
- The evidence agent is deterministic, token-free artifact reporting; it is not an LLM.
- Detector latency excludes ingestion, packet parsing, and report generation.
- Some rare classes have very small held-out support, especially in the NF-UQ-NIDS-v2 minority challenge.
- The exact historical source snapshot associated with every remote run is unavailable; the retained code and evidence are therefore documented without claiming byte-identical reconstruction.

## Repository policy

Do not commit raw datasets, credentials, private manuscript-review reports, local cache files, or unsupported generated results. Preserve failed runs and negative results rather than rewriting history to improve a comparison.

