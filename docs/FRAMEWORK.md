# ATLAS-IDNet Framework

ATLAS-IDNet is a multi-branch intrusion-detection framework for tabular network-flow benchmarks.

## Model

The neural detector splits engineered flow features into traffic, protocol, and temporal branches. Each branch learns a dense representation, then an attention fusion layer weights the branches before classification.

## Paper-Facing Model

The active paper-facing implementation focuses on ATLAS-IDNet only. Classical machine-learning baselines are disabled in the main training notebook so the framework is positioned around the proposed deep detector rather than a mixed ML benchmark suite.

## Ablation

The ablation stage should be interpreted as feature-branch evidence for the proposed detector. It evaluates:

- traffic-only features
- protocol-only features
- temporal-only features
- traffic plus protocol
- traffic plus temporal
- all features

## Explainability

The explainability layer writes:

- random forest feature importance
- permutation importance on a held-out sample
- feature-group importance
- ATLAS-IDNet branch-attention summaries

## Agentic Layer

The agentic report reads saved EDA, metric, ablation, latency, and explanation artifacts. It creates a grounded Markdown and JSON report for each dataset/task without fabricating unavailable metrics.
