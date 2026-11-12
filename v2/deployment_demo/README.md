# ATLAS-IDNet Local Deployment Demonstrator

This one-page console replays the verified seed-42 natural-prior diagnostic
cohort for CICIoT2023 and NF-UQ-NIDS-v2. It loads the downloaded PyTorch
checkpoints on CPU, compares local probabilities with stored held-out
evaluation probabilities, and renders the measured latency, resource,
explanation, calibration, and deterministic-agent artifacts.

## Run

From the project root:

```bash
PYTHONPATH=implementation/v2/src \
  /Users/darsh/miniconda3/envs/research/bin/python \
  implementation/v2/deployment_demo/server.py --port 8766
```

Open <http://127.0.0.1:8766>.

## Verification boundary

- The page uses real downloaded checkpoints and immutable result artifacts.
- Replay inputs are the 256 standardized held-out flows used by the verified
  occlusion diagnostics; this is not live packet capture.
- The stored p50/p95/p99 and throughput values come from `latency.csv`.
- Per-request replay time is displayed separately and is not substituted for
  the controlled benchmark.
- The evidence agent is deterministic and token-free; it is not an LLM.
- The screenshots are candidate paper assets. They are not inserted into the
  manuscript by this task.

## Publication screenshots

With the server running:

```bash
NODE_PATH=/Users/darsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  /Users/darsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  implementation/v2/deployment_demo/capture_screenshots.cjs
```

The script performs desktop/mobile overflow checks, validates canvas sizing,
fails on browser errors, and writes 2x-resolution PNG files to `screenshots/`.

### Recommended complete-page captures

- `atlas_deployment_console_cic.png`: complete CICIoT2023 console.
- `atlas_deployment_console_nf.png`: complete NF-UQ-NIDS-v2 console.

### Explainability overview captures

- `atlas_explainability_overview_cic.png`: all four CICIoT2023 explanation
  views in one section.
- `atlas_explainability_overview_nf.png`: all four NF-UQ-NIDS-v2 explanation
  views in one section.

### Separate explainability figures

Each file is 2400 pixels wide and includes its own title and measurement scope:

- `atlas_feature_evidence_{cic,nf}.png`;
- `atlas_branch_attention_{cic,nf}.png`;
- `atlas_faithfulness_{cic,nf}.png`;
- `atlas_calibration_{cic,nf}.png`.

`qa_mobile.png` is retained only as responsive-layout evidence. The compact
profile crop is optional; the complete-page captures already include the same
deployment evidence in context.
