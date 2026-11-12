from __future__ import annotations

from datetime import datetime, timezone
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from atlas_idnet.config import write_json
from atlas_idnet.models.multimodal import parameter_count, predict_proba


def measure_latency(model, x_test: np.ndarray, output_dir: Path, samples: int = 4096) -> dict:
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    x = x_test[: min(samples, len(x_test))]
    if len(x) == 0:
        raise ValueError("No samples available for latency measurement")
    device = str(next(model.parameters()).device)
    predict_proba(model, x[: min(64, len(x))])
    start = time.perf_counter()
    predict_proba(model, x)
    elapsed = time.perf_counter() - start
    per_sample_ms = elapsed / len(x) * 1000.0
    batch_rows = []
    for batch_size in [1, 8, 32, 128, 512, 2048]:
        if batch_size > len(x):
            continue
        repetitions = min(50, max(5, len(x) // batch_size))
        durations = []
        for repeat in range(repetitions):
            start_idx = (repeat * batch_size) % max(1, len(x) - batch_size + 1)
            batch = x[start_idx : start_idx + batch_size]
            t0 = time.perf_counter()
            predict_proba(model, batch, batch_size=batch_size)
            durations.append(time.perf_counter() - t0)
        per_sample = np.asarray(durations) / batch_size * 1000.0
        batch_rows.append(
            {
                "batch_size": batch_size,
                "repetitions": repetitions,
                "latency_ms_mean": float(per_sample.mean()),
                "latency_ms_p50": float(np.percentile(per_sample, 50)),
                "latency_ms_p95": float(np.percentile(per_sample, 95)),
                "latency_ms_p99": float(np.percentile(per_sample, 99)),
                "throughput_samples_per_second_mean": float(batch_size / np.asarray(durations).mean()),
            }
        )
    pd.DataFrame(batch_rows).to_csv(output_dir / "latency_by_batch.csv", index=False)
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "model_parameters": parameter_count(model),
        "feature_dim": int(x.shape[1]) if x.ndim == 2 else None,
        "samples": int(len(x)),
        "total_seconds": elapsed,
        "latency_ms_per_sample": per_sample_ms,
        "throughput_samples_per_second": len(x) / elapsed if elapsed else None,
        "batch_latency_table": "latency_by_batch.csv",
    }
    write_json(output_dir / "latency.json", result)
    return result
