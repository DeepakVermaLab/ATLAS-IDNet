#!/usr/bin/env python3
"""Local ATLAS-IDNet deployment and evidence demonstrator.

The service loads immutable seed-42 natural-prior checkpoints and replays the
held-out diagnostic cohort used by the verified explanation analysis. It does
not emulate packet capture or alter any controlled-experiment artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
V2_ROOT = ROOT.parent
SRC_ROOT = V2_ROOT / "src"
STATIC_ROOT = ROOT / "static"
BUNDLES_ROOT = ROOT / "bundles"
sys.path.insert(0, str(SRC_ROOT))

from atlas_v2.models import Detector  # noqa: E402


DATASETS = {
    "cic": {
        "name": "CICIoT2023",
        "protocol": "Natural prior",
        "run": "atlas_balanced_42_a66ef4d06f327ab2",
    },
    "nf": {
        "name": "NF-UQ-NIDS-v2",
        "protocol": "Natural prior",
        "run": "atlas_balanced_42_31809d9749ba9034",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_float(value):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value}")
    return number


class DatasetRuntime:
    def __init__(self, key: str):
        self.key = key
        self.spec = DATASETS[key]
        self.root = BUNDLES_ROOT / key
        self.config = read_json(self.root / "config.json")
        self.metadata = read_json(self.root / "metadata.json")
        self.metrics = read_json(self.root / "metrics.json")
        self.architecture = read_json(self.root / "architecture.json")
        self.resource = read_json(self.root / "resource_profile.json")
        self.stability = read_json(self.root / "stability.json")
        self.explanation_scope = read_json(self.root / "explanation_scope.json")
        self.agent = read_json(self.root / "agent" / "agent_report.json")
        self.latency = read_csv(self.root / "latency.csv")
        self.faithfulness = read_csv(self.root / "faithfulness.csv")
        self.attention_summary = read_csv(self.root / "attention_summary.csv")
        self.calibration = read_csv(self.root / "test_calibration.csv")

        self.occlusion = np.load(self.root / "occlusion_evidence.npz", allow_pickle=False)
        self.predictions = np.load(self.root / "test_predictions.npz", allow_pickle=False)
        self.features = [str(value) for value in self.metadata["features"]]
        self.classes = [str(value) for value in self.metadata["classes"]]

        self.model = Detector(
            self.features,
            len(self.classes),
            self.config["variant"],
            self.config["width"],
            self.config["seed"],
            self.config.get("dropout", 0.15),
        )
        self.model.load_state_dict(
            torch.load(self.root / "best.pt", weights_only=True, map_location="cpu")
        )
        self.model.eval()
        torch.set_num_threads(1)
        self._verify_bundle()

    def _verify_bundle(self):
        x = self.occlusion["x"].astype("float32")
        test_indices = self.occlusion["test_indices"].astype(int)
        with torch.inference_mode():
            local = self.model(torch.from_numpy(x)).softmax(1).numpy()
        stored = self.predictions["proba"][test_indices]
        self.max_probability_delta = float(np.max(np.abs(local - stored)))
        self.prediction_agreement = float(
            np.mean(local.argmax(1) == stored.argmax(1))
        )
        if self.max_probability_delta > 1e-5 or self.prediction_agreement != 1.0:
            raise ValueError(
                f"Bundle verification failed for {self.spec['name']}: "
                f"delta={self.max_probability_delta}, "
                f"agreement={self.prediction_agreement}"
            )
        if self.agent["verification"]["unsupported_claim_rate"] != 0:
            raise ValueError(f"Agent report is not fully supported for {self.spec['name']}")

    def _latency_row(self, batch: int):
        for row in self.latency:
            if int(float(row["batch"])) == batch:
                return row
        raise KeyError(f"Missing latency batch {batch}")

    def summary(self):
        single = self._latency_row(1)
        maximum_throughput = max(
            clean_float(row["throughput_flows_s"]) for row in self.latency
        )
        calibration = [
            {
                "confidence": clean_float(row["confidence"]),
                "accuracy": clean_float(row["accuracy"]),
                "count": int(float(row["count"])),
            }
            for row in self.calibration
        ]
        return {
            "key": self.key,
            "dataset": self.spec["name"],
            "protocol": self.spec["protocol"],
            "run": self.spec["run"],
            "seed": int(self.config["seed"]),
            "variant": self.config["variant"],
            "classes": len(self.classes),
            "features": len(self.features),
            "diagnostic_samples": int(len(self.occlusion["x"])),
            "metrics": {
                key: clean_float(self.metrics[key])
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "mcc",
                    "ece_15",
                )
            },
            "latency": {
                "p50_single_ms": clean_float(single["p50_batch_ms"]),
                "p95_single_ms": clean_float(single["p95_batch_ms"]),
                "p99_single_ms": clean_float(single["p99_batch_ms"]),
                "maximum_throughput_flows_s": maximum_throughput,
                "scope": single["scope"],
            },
            "resource": {
                "parameters": int(self.resource["trainable_parameters"]),
                "model_file_bytes": int(self.resource["model_file_bytes"]),
                "flops_per_flow": int(self.resource["linear_flops_per_flow"]),
                "peak_process_rss_bytes": int(
                    self.resource["process_peak_rss_after_diagnostics_bytes"]
                ),
                "device": self.resource["device"],
                "threads": int(self.resource["threads"]),
            },
            "attention": [
                {
                    "branch": row["branch"],
                    "mean": clean_float(row["mean"]),
                    "std": clean_float(row["std"]),
                }
                for row in self.attention_summary
            ],
            "faithfulness": [
                {
                    "fraction": clean_float(row["fraction"]),
                    "ranked_drop": clean_float(row["top_deletion_drop"]),
                    "random_drop": clean_float(row["random_deletion_drop"]),
                    "sufficiency_gap": clean_float(row["sufficiency_gap"]),
                }
                for row in self.faithfulness
            ],
            "stability": {
                "prediction_agreement": clean_float(
                    self.stability["prediction_agreement"]
                ),
                "feature_overlap": clean_float(
                    self.stability["top_quarter_jaccard_like_overlap"]
                ),
                "noise_std": clean_float(self.stability["noise_std_standardized"]),
            },
            "calibration": calibration,
            "agent": {
                "name": self.agent["agent"],
                "findings": self.agent["findings"],
                "latency_ms": clean_float(self.agent["latency_ms"]),
                "accepted": int(self.agent["verification"]["accepted"]),
                "unsupported_claim_rate": clean_float(
                    self.agent["verification"]["unsupported_claim_rate"]
                ),
                "limitations": self.agent["limitations"],
            },
            "verification": {
                "max_probability_delta": self.max_probability_delta,
                "prediction_agreement": self.prediction_agreement,
                "checkpoint_sha256": sha256(self.root / "best.pt"),
                "scope": "Local CPU replay of the held-out diagnostic cohort",
            },
            "limitations": {
                "replay": "Prepared held-out flows; not live packet capture.",
                "explanation": self.explanation_scope["limitations"],
                "deployment": "Measured on one CPU environment; no universal deployment claim.",
            },
        }

    def sample(self, index: int):
        count = len(self.occlusion["x"])
        position = index % count
        x = self.occlusion["x"][position : position + 1].astype("float32")
        impact = self.occlusion["impact"][position]
        test_index = int(self.occlusion["test_indices"][position])

        started = time.perf_counter_ns()
        with torch.inference_mode():
            logits, attention = self.model(torch.from_numpy(x), return_attention=True)
            probability = logits.softmax(1).numpy()[0]
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6

        stored = self.predictions["proba"][test_index]
        true_index = int(self.predictions["y_true"][test_index])
        prediction = int(probability.argmax())
        order = np.argsort(-probability)[:5]
        evidence_order = np.argsort(-np.abs(impact))[:8]
        branch_names = [row["branch"] for row in self.attention_summary]
        source_identifier = str(self.predictions["ids"][test_index])
        identifier = "flow-" + hashlib.sha256(source_identifier.encode()).hexdigest()[:12]
        return {
            "position": position,
            "sample_count": count,
            "event_id": identifier,
            "test_index": test_index,
            "true_label": self.classes[true_index],
            "predicted_label": self.classes[prediction],
            "correct": prediction == true_index,
            "confidence": clean_float(probability[prediction]),
            "local_replay_ms": elapsed_ms,
            "stored_probability_delta": float(np.max(np.abs(probability - stored))),
            "top_probabilities": [
                {"label": self.classes[int(i)], "probability": clean_float(probability[i])}
                for i in order
            ],
            "attention": [
                {"branch": name, "weight": clean_float(attention[0, i])}
                for i, name in enumerate(branch_names)
            ],
            "feature_evidence": [
                {
                    "feature": self.features[int(i)],
                    "impact": clean_float(impact[i]),
                    "standardized_value": clean_float(x[0, i]),
                }
                for i in evidence_order
            ],
            "scope": "Held-out diagnostic replay; local timing is illustrative, not the benchmark statistic.",
        }


RUNTIMES = {key: DatasetRuntime(key) for key in DATASETS}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json({"datasets": {key: runtime.summary() for key, runtime in RUNTIMES.items()}})
            return
        if parsed.path == "/api/sample":
            query = parse_qs(parsed.query)
            key = query.get("dataset", ["cic"])[0]
            if key not in RUNTIMES:
                self._json({"error": "Unknown dataset"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                index = int(query.get("index", ["0"])[0])
            except ValueError:
                self._json({"error": "Index must be an integer"}, HTTPStatus.BAD_REQUEST)
                return
            self._json(RUNTIMES[key].sample(index))
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"ATLAS-IDNet deployment console: http://{args.bind}:{args.port}", flush=True)
    print("Verified bundles: " + ", ".join(runtime.spec["name"] for runtime in RUNTIMES.values()), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
