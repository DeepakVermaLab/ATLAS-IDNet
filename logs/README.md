# Execution evidence

The repository does not manufacture a consolidated log after the fact. Execution evidence is preserved where it was originally produced:

- printed cell output in the executed notebooks under `notebooks/`, `download/notebooks/`, `upload/notebooks/`, and selected `v2/notebooks/`;
- `results/scaffold_environment_check/logs/environment.log`;
- immutable completion and diagnostic artifacts under `v2/deployment_demo/bundles/`;
- run ledgers, confidence summaries, and paired tests under `evidence/v2/`.

Some v2 notebooks in this local snapshot have cleared execution counters even though durable remote artifacts prove completion. They must not be presented as executed-notebook evidence; `v2/EXECUTION_STATUS.md` and the retained artifacts are authoritative for those runs.

