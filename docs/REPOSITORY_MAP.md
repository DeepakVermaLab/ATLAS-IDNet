# Repository map

This repository is organized as the implementation root. It contains two related code lines because both are necessary to understand the project history and reproduce the final controlled evaluation.

## Original end-to-end implementation

```text
configs/       Runtime configuration
download/      Dataset acquisition notebooks and helpers
upload/        Drive-transfer notebooks and helpers
notebooks/     Executed end-to-end notebooks from setup through reporting
scripts/       Command-line entry points matching the notebook stages
src/           Original atlas_idnet Python package
results/       Retained environment/log artifacts
docs/          Framework and operational documentation
```

## Controlled v2 implementation

```text
v2/notebooks/        Pilot and publication-protocol notebooks
v2/src/atlas_v2/     Data preparation, detector, experiments, and evidence agent
v2/tests/            Sampling, rare-split, and full synthetic smoke tests
v2/deployment_demo/  Private local CPU inference demonstrator and verified bundles
v2/EXECUTION_STATUS.md
```

## Evidence snapshot

`evidence/v2/` contains 75 selected files copied from the verified result tree for manuscript and repository audit. It is intentionally smaller than the full remote artifact tree. It includes aggregate ledgers and representative seed-42 diagnostic files but not raw datasets or every model checkpoint.

## Generated and excluded content

The repository excludes raw datasets, AppleDouble metadata, Python bytecode, cache directories, duplicate ZIP archives, rendered documentation previews, and private manuscript-quality reports. Dataset paths in historical notebooks and manifests document the original environment; users should replace them through the configuration cells or files rather than editing model code.

