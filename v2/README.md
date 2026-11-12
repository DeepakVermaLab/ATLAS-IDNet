# ATLAS-IDNet implementation v2

This directory is isolated from the submitted implementation and manuscript.
It executes real-data validation and preserves every completed or negative run.

## Notebook order

The completed pilot notebooks remain unchanged. The publication-oriented
extension starts at notebook 04:

1. `00_setup_and_paths.ipynb`: original environment check.
2. `01_prepare_and_cpu_pilot.ipynb`: completed 100,000-row CPU pilot.
3. `04_prepare_publication_protocols.ipynb`: prepares a 500,000-row natural-prior
   sample and a separate 5,000-record-per-class minority challenge for each
   dataset. Class capping changes priors and is never substituted for the
   natural-distribution result.
4. `05_publication_controlled_experiments.ipynb`: repeated-seed proposed model,
   faithful legacy detector, parameter-matched component controls, and four
   classical baselines.
5. `06_diagnostics_and_evidence_agent.ipynb`: single-thread CPU latency,
   explanation faithfulness/stability, attention evidence, seed statistics,
   paired tests, artifact-grounded agent evaluation, and final implementation
   audit.

Notebooks 02 and 03 belong to the earlier compact pilot protocol and are not
the publication-extension execution path.

## Compute policy

CPU is the default and is sufficient for preparation, baselines, repeated
training, statistics, latency, explanations, and the evidence agent. Neural
checkpoints contain device-independent state dictionaries. When a GPU is
available, `device="auto"` uses it; a later CPU session can resume the same
checkpoint. GPU allocation loss disconnects the runtime, so completion still
depends on the latest checkpoint having reached Drive.

## Evidence and controls

- Train-only median imputation, clipping and standardization.
- Source SHA-256 values, row identifiers, deduplication and partition-overlap
  audits.
- Fixed five-seed protocol with validation macro-F1 early stopping.
- Proposed semantic-branch detector with residual encoders, attention fusion
  and a gated global context path.
- Faithful submitted-topology `legacy` comparator.
- Capacity-matched `no_global_context`, `no_residual`, `mean`, `concat`,
  `random_groups`, `single_residual`, and `mlp` controls.
- Random forest, extra trees, histogram gradient boosting and logistic
  regression baselines.
- Accuracy, balanced accuracy, precision, recall, macro/weighted F1, MCC,
  Cohen's kappa, log loss, multiclass Brier score, ECE, supported AUROC, and
  benign-versus-intrusion error rates.
- Predictions, per-class reports, confusion matrices, calibration bins,
  training histories, architecture manifests, checkpoints, environments,
  confidence intervals and paired statistical tests.
- CPU p50/p95/p99 latency, throughput, model size and linear-operation estimate.
- Occlusion deletion, random-deletion control, sufficiency, perturbation
  stability and branch-attention summaries.
- A tool-using finite-state evidence agent whose claims must reference measured
  artifacts. It is not an LLM and makes no LLM claim.

## Scientific boundaries

The previous submitted scores are stored only as descriptive targets. They are
not independently reproduced baselines and cannot be used to tune on the test
set. A stronger ablation is retained and reported; it is never weakened or
removed. The proposed detector can be changed only from training/validation
evidence, followed by a complete matched rerun.

The automatic feature grouping is recorded for domain review. Temporal features
are time-derived tabular statistics, not a temporal sequence. The random split
does not establish cross-site or temporal deployment generalization. Grouped or
temporal evaluation requires genuine metadata and must never be synthesized
from row order.

Synthetic tests validate code only. They are not research results. Journal
acceptance and metric superiority are not guaranteed by this implementation.
