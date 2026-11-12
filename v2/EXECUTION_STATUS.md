# Implementation execution status

Last verified: 2026-10-23 22:59 IST

Scope: implementation and measured experiments only. Manuscript writing or
revision was not performed.

## Status

The ATLAS-IDNet v2 implementation and real-data verification phase is complete.
Notebook 05 remains complete and was not rerun during the final diagnostics
recovery. Notebook 06 completed the four seed-42 CPU diagnostic evaluations.

## Verified results inventory

- Publication runs: 180/180 valid `complete.json` records.
- CICIoT2023 natural: 65/65.
- CICIoT2023 minority challenge: 25/25.
- NF-UQ-NIDS-v2 natural: 65/65.
- NF-UQ-NIDS-v2 minority challenge: 25/25.
- Neural runs: 130; classical runs: 50.
- Drive inventory: 3,248 unique, nonempty files; zero duplicate paths.
- Four protocol roots contain `all_runs.csv`, `seed_summary.csv`, and
  `paired_tests.csv` with the expected variant/seed matrices and consistent
  data/source hashes.
- Every completed run contains architecture, environment, metrics, per-class,
  prediction, confusion, and calibration artifacts plus its model artifact.
- Four seed-42 ATLAS runs contain latency, faithfulness, random-control,
  sufficiency, stability, attention, explanation-scope, occlusion, and CPU
  resource-profile artifacts.
- All four `resource_profile.json` files record CPU execution with one thread
  and positive RSS, model-size, parameter, MAC/FLOP, and input-size values.
- Agent evaluation contains four complete reports, three findings per report,
  and zero unsupported claims.
- `final_implementation_audit.csv` has four protocol rows and agrees with the
  means recomputed from `all_runs.csv`.

## Scientific boundaries

- Natural-prior and class-capped minority-challenge results remain separate.
- Held-out class coverage is present in all four prepared protocols with zero
  partition row-ID overlap. Very small natural-prior held-out supports remain a
  limitation and were not removed.
- Strong classical controls outperform ATLAS in several comparisons. These
  negative results are preserved; no ablation was weakened and no test-set
  tuning was performed.
- Only CICIoT2023 natural exceeds the older submitted-reference accuracy and
  weighted-F1 targets. The other three protocol-level comparisons do not.
- The evidence agent is deterministic and artifact-grounded, not an LLM.
- Random stratified evaluation does not establish temporal, site, device, or
  deployment generalization. Occlusion evidence does not establish causality.

## Verification

The following local tests passed after the resource-profiling change:

```bash
PYTHONPATH=implementation/v2/src \
  /Users/darsh/miniconda3/envs/research/bin/python implementation/v2/tests/test_sampling.py
PYTHONPATH=implementation/v2/src \
  /Users/darsh/miniconda3/envs/research/bin/python implementation/v2/tests/test_rare_split.py
PYTHONPATH=implementation/v2/src \
  /Users/darsh/miniconda3/envs/research/bin/python implementation/v2/tests/smoke.py
```

Drive source parity, excluding the two output-bearing notebooks, reported 18
matching files and zero differences. No file under `paper/` was modified during
the final implementation run.

## Next boundary

Stop here. Do not start manuscript work unless the user explicitly authorizes a
new manuscript phase after reading the verified implementation outcome.
