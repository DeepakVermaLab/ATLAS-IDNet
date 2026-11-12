#!/usr/bin/env python3
"""Export a de-identified recorded replay for the public static simulator."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "v2" / "deployment_demo" / "server.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("atlas_deployment_server", SERVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load deployment runtime: {SERVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sanitized_summary(runtime):
    summary = runtime.summary()
    summary["agent"]["findings"] = [
        {
            "claim_type": finding["claim_type"],
            "statement": finding["statement"],
            "values": finding.get("values", {}),
        }
        for finding in summary["agent"]["findings"]
    ]
    summary["verification"]["scope"] = (
        "Recorded local CPU replay of the held-out diagnostic cohort"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    module = load_runtime_module()
    datasets = {}
    for key, runtime in module.RUNTIMES.items():
        samples = []
        for index in range(len(runtime.occlusion["x"])):
            sample = runtime.sample(index)
            sample["scope"] = (
                "Recorded held-out diagnostic replay; export-time CPU timing is "
                "illustrative and is not the controlled benchmark statistic."
            )
            samples.append(sample)
        datasets[key] = {
            "summary": sanitized_summary(runtime),
            "samples": samples,
        }

    payload = {
        "schema_version": 1,
        "mode": "recorded-replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "boundary": (
            "De-identified recorded evidence only; no live packet capture, "
            "browser-side model inference, autonomous response, or LLM is used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(
        f"Wrote {args.output} with "
        f"{sum(len(value['samples']) for value in datasets.values())} samples"
    )


if __name__ == "__main__":
    main()

