"""Generate publication-protocol notebooks without modifying completed pilot notebooks."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

BOOTSTRAP = '''from pathlib import Path
import os, sys, json, errno, time
DRIVE = Path("/content/drive/MyDrive")
try:
    from google.colab import drive
except ImportError:
    drive = None

def with_drive_retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            disconnected = exc.errno in {errno.ENOTCONN, errno.EIO, errno.ESTALE} or "Transport endpoint" in str(exc)
            if not disconnected or attempt == attempts - 1:
                raise
            print(f"Drive disconnected; remounting before retry {attempt + 1}/{attempts - 1}.", flush=True)
            if drive is not None:
                drive.mount("/content/drive", force_remount=True)
            time.sleep(1)

def ensure_drive():
    if drive is not None:
        if not DRIVE.is_dir():
            drive.mount("/content/drive", force_remount=False)
        next(DRIVE.iterdir(), None)
with_drive_retry(ensure_drive)

def resolve_v2():
    configured = os.environ.get("ATLAS_V2_ROOT")
    candidates = ([Path(configured).expanduser()] if configured else []) + [
        DRIVE/"projects/implementation/ATLAS-IDNet/implementation/v2",
        Path.cwd(), *Path.cwd().parents,
    ]
    for candidate in candidates:
        if all((candidate/name).is_file() for name in (
            "src/atlas_v2/data.py", "src/atlas_v2/models.py",
            "src/atlas_v2/experiments.py", "src/atlas_v2/agent.py", "requirements.txt"
        )):
            return candidate.resolve()
    raise FileNotFoundError("Synchronized implementation/v2 source is not visible to this kernel.")

V2 = with_drive_retry(resolve_v2)
sys.path.insert(0, str(V2/"src"))
for module in list(sys.modules):
    if module == "atlas_v2" or module.startswith("atlas_v2."):
        del sys.modules[module]
import atlas_v2
print("Code:", V2)
'''

SETUP = '''import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(V2/"requirements.txt")])
import numpy as np, pandas as pd, sklearn, scipy, torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU count:", os.cpu_count())
'''

LOAD = '''RESULTS = DRIVE/"projects/implementation/ATLAS-IDNet/results/v2" if DRIVE.exists() else V2/"results"
PUBLICATION_CONFIG_PATH = RESULTS/"publication_config.json"
PUBLICATION_CONFIG = json.loads(PUBLICATION_CONFIG_PATH.read_text())
print(json.dumps(PUBLICATION_CONFIG, indent=2))
'''

CONFIGURE_PROTOCOLS = '''BASE = DRIVE/"projects/Dataset/Security" if DRIVE.exists() else Path("/Volumes/LocalDrive1/Dataset/Security")
RESULTS = DRIVE/"projects/implementation/ATLAS-IDNet/results/v2" if DRIVE.exists() else V2/"results"
RESULTS.mkdir(parents=True, exist_ok=True)
common = {
    "split": "stratified", "group_column": None, "time_column": None,
    "exclude_columns": ["Label", "label", "Attack", "Attack_type", "Class", "Dataset", "dataset",
                        "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Flow ID", "Src IP", "Dst IP", "Timestamp"]
}
datasets = {
    "CICIoT2023": {"csv_globs": [str(BASE/"CICIoT2023/**/*.csv")], "label": "label"},
    "NF-UQ-NIDS-v2": {
        "csv_globs": [str(BASE/"NF-UQ-NIDS-v2/**/*.csv")],
        "label": "Attack",
        "conflicting_duplicate_policy": "drop_all",
    },
}
protocols = {}
for dataset_name, dataset in datasets.items():
    natural = {**dataset, **common, "max_rows": 500000}
    minority = {**dataset, **common, "max_rows": 500000, "class_cap": 5000}
    protocols[dataset_name] = {"natural": natural, "minority_challenge": minority}
PUBLICATION_CONFIG = {
    "results": str(RESULTS),
    "seeds": [42, 43, 44, 45, 46],
    "challenge_seeds": [42, 43, 44, 45, 46],
    "epochs": 30,
    "width": 128,
    "batch_size": 512,
    "patience": 7,
    "loss": "balanced",
    "challenge_support_policy": {
        "required_minimum_per_heldout_partition": 15,
        "preferred_minimum_per_heldout_partition": 30,
        "below_preferred_interpretation": "source-limited; retain the class, report uncertainty, and avoid strong class-specific claims",
    },
    "protocols": protocols,
    "submitted_reference": {
        "CICIoT2023": {"accuracy": 0.9503, "weighted_f1": 0.9462, "macro_f1": 0.6460},
        "NF-UQ-NIDS-v2": {"accuracy": 0.9795, "weighted_f1": 0.9787, "macro_f1": 0.7154},
        "provenance": "Previously submitted manuscript; target only, not an independently reproduced baseline."
    }
}
PUBLICATION_CONFIG_PATH = RESULTS/"publication_config.json"
temporary = PUBLICATION_CONFIG_PATH.with_suffix(".tmp")
temporary.write_text(json.dumps(PUBLICATION_CONFIG, indent=2))
os.replace(temporary, PUBLICATION_CONFIG_PATH)
print("Publication config:", PUBLICATION_CONFIG_PATH)
print(json.dumps(PUBLICATION_CONFIG, indent=2))
'''

PREPARE_PROTOCOLS = '''from atlas_v2.data import PREPARATION_VERSION, prepare
import hashlib
support_policy = PUBLICATION_CONFIG["challenge_support_policy"]
PUBLICATION_CONFIG["preparation_support_audit"] = {}
for dataset_name, dataset_protocols in PUBLICATION_CONFIG["protocols"].items():
    for protocol_name, spec in dataset_protocols.items():
        clean_spec = {key: value for key, value in spec.items() if key != "prepared"}
        key = hashlib.sha256(json.dumps({"spec": clean_spec, "version": PREPARATION_VERSION}, sort_keys=True).encode()).hexdigest()[:12]
        folder = RESULTS/dataset_name/f"prepared_{protocol_name}_{key}"
        if not (folder/"metadata.json").exists():
            with_drive_retry(lambda: prepare(clean_spec, folder, seed=42))
        metadata = json.loads((folder/"metadata.json").read_text())
        PUBLICATION_CONFIG["protocols"][dataset_name][protocol_name]["prepared"] = str(folder)
        print(dataset_name, protocol_name, metadata["counts"])
        if protocol_name == "minority_challenge" and not metadata["full_class_evaluation_supported"]:
            raise RuntimeError(f"{dataset_name}: minority challenge lacks held-out support: {metadata['missing_classes']}")
        if protocol_name == "minority_challenge":
            minimum = metadata["minimum_heldout_support"]
            required = support_policy["required_minimum_per_heldout_partition"]
            preferred = support_policy["preferred_minimum_per_heldout_partition"]
            if minimum < required:
                raise RuntimeError(
                    f"{dataset_name}: minority challenge has only {minimum} records in its least-supported "
                    f"held-out class, below the explicit minimum of {required}."
                )
            status = "adequate" if minimum >= preferred else "source_limited"
            PUBLICATION_CONFIG["preparation_support_audit"][f"{dataset_name}/{protocol_name}"] = {
                "minimum_heldout_support": minimum,
                "required_minimum": required,
                "preferred_minimum": preferred,
                "status": status,
                "interpretation": ("standard challenge support" if status == "adequate" else
                    support_policy["below_preferred_interpretation"]),
            }
            if status == "source_limited":
                print(
                    f"WARNING: {dataset_name} minority challenge is source-limited at {minimum} held-out records; "
                    "the class is retained, but its class-specific estimate requires uncertainty reporting.",
                    flush=True,
                )
        temporary = PUBLICATION_CONFIG_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(PUBLICATION_CONFIG, indent=2))
        os.replace(temporary, PUBLICATION_CONFIG_PATH)
'''

AUDIT_PREPARATION = '''audit_rows = []
support_policy = PUBLICATION_CONFIG["challenge_support_policy"]
for dataset_name, dataset_protocols in PUBLICATION_CONFIG["protocols"].items():
    for protocol_name, spec in dataset_protocols.items():
        metadata = json.loads((Path(spec["prepared"])/"metadata.json").read_text())
        audit_rows.append({
            "dataset": dataset_name, "protocol": protocol_name,
            "rows_scanned": metadata["source_rows_scanned"],
            "retained_rows": sum(sum(values) for values in metadata["counts"].values()),
            "features": len(metadata["features"]), "classes": len(metadata["classes"]),
            "duplicates_removed": metadata["duplicates_removed"],
            "ambiguous_duplicate_groups": metadata.get("conflicting_duplicates", 0),
            "ambiguous_rows_removed": metadata.get("ambiguous_rows_removed", 0),
            "minimum_heldout_support": metadata["minimum_heldout_support"],
            "required_minimum_heldout_support": (support_policy["required_minimum_per_heldout_partition"]
                                                  if protocol_name == "minority_challenge" else None),
            "preferred_minimum_heldout_support": (support_policy["preferred_minimum_per_heldout_partition"]
                                                   if protocol_name == "minority_challenge" else None),
            "support_status": (PUBLICATION_CONFIG.get("preparation_support_audit", {})
                               .get(f"{dataset_name}/{protocol_name}", {}).get("status", "natural_prior")),
            "full_class_evaluation_supported": metadata["full_class_evaluation_supported"],
            "sampling_protocol": metadata["sampling_protocol"],
        })
audit = pd.DataFrame(audit_rows)
audit.to_csv(RESULTS/"preparation_audit.csv", index=False)
display(audit)
'''

BUILD_PLAN = '''from atlas_v2.models import Detector, matched_width, parameter_count
plan = []
for dataset_name, dataset_protocols in PUBLICATION_CONFIG["protocols"].items():
    natural_meta = json.loads((Path(dataset_protocols["natural"]["prepared"])/"metadata.json").read_text())
    reference = Detector(natural_meta["features"], len(natural_meta["classes"]), "atlas", PUBLICATION_CONFIG["width"])
    target = parameter_count(reference)
    for variant in ["atlas", "legacy", "no_global_context", "no_residual", "mean", "concat", "random_groups", "single_residual", "mlp"]:
        if variant in {"atlas", "legacy"}:
            selected_width = PUBLICATION_CONFIG["width"]
            count = parameter_count(Detector(natural_meta["features"], len(natural_meta["classes"]), variant, selected_width))
        else:
            match = matched_width(natural_meta["features"], len(natural_meta["classes"]), variant, target)
            selected_width, count = match["width"], match["parameters"]
        plan.append({"dataset": dataset_name, "protocol": "natural", "variant": variant,
                     "width": selected_width, "parameters": count, "target_parameters": target,
                     "seeds": len(PUBLICATION_CONFIG["seeds"])})
    for variant in ["atlas", "legacy", "no_global_context", "mean", "extra_trees"]:
        plan.append({"dataset": dataset_name, "protocol": "minority_challenge", "variant": variant,
                     "width": PUBLICATION_CONFIG["width"], "parameters": None, "target_parameters": target,
                     "seeds": len(PUBLICATION_CONFIG["challenge_seeds"])})
plan = pd.DataFrame(plan)
plan.to_csv(RESULTS/"experiment_plan.csv", index=False)
display(plan)
'''

RUN_EXPERIMENTS = '''from atlas_v2.experiments import run
from atlas_v2.models import Detector, parameter_count

def run_resilient(*args, **kwargs):
    try:
        return run(*args, **kwargs)
    except RuntimeError as error:
        if kwargs.get("device") == "auto" and ("CUDA" in str(error) or "cuda" in str(error)):
            print("CUDA run failed; resuming the same portable checkpoint on CPU:", error, flush=True)
            kwargs["device"] = "cpu"
            return run(*args, **kwargs)
        raise

for dataset_name, dataset_protocols in PUBLICATION_CONFIG["protocols"].items():
    natural = dataset_protocols["natural"]
    metadata = json.loads((Path(natural["prepared"])/"metadata.json").read_text())
    target = parameter_count(Detector(metadata["features"], len(metadata["classes"]), "atlas", PUBLICATION_CONFIG["width"]))
    root = RESULTS/dataset_name/"publication"/"natural"
    for variant in ["atlas", "legacy", "no_global_context", "no_residual", "mean", "concat", "random_groups", "single_residual", "mlp"]:
        capacity_target = None if variant in {"atlas", "legacy"} else target
        for seed in PUBLICATION_CONFIG["seeds"]:
            run_resilient(natural["prepared"], root, variant=variant, seed=seed,
                          epochs=PUBLICATION_CONFIG["epochs"], width=PUBLICATION_CONFIG["width"],
                          batch_size=PUBLICATION_CONFIG["batch_size"], patience=PUBLICATION_CONFIG["patience"],
                          loss=PUBLICATION_CONFIG["loss"], device="auto", capacity_target=capacity_target)
    for variant in ["random_forest", "extra_trees", "hist_gradient", "logistic"]:
        for seed in PUBLICATION_CONFIG["seeds"]:
            run_resilient(natural["prepared"], root, variant=variant, seed=seed,
                          epochs=PUBLICATION_CONFIG["epochs"], loss=PUBLICATION_CONFIG["loss"], device="cpu")

    challenge = dataset_protocols["minority_challenge"]
    challenge_root = RESULTS/dataset_name/"publication"/"minority_challenge"
    for variant in ["atlas", "legacy", "no_global_context", "mean", "extra_trees"]:
        for seed in PUBLICATION_CONFIG["challenge_seeds"]:
            run_resilient(challenge["prepared"], challenge_root, variant=variant, seed=seed,
                          epochs=PUBLICATION_CONFIG["epochs"], width=PUBLICATION_CONFIG["width"],
                          batch_size=PUBLICATION_CONFIG["batch_size"], patience=PUBLICATION_CONFIG["patience"],
                          loss=PUBLICATION_CONFIG["loss"], device="auto")
'''

SUMMARIZE = '''from atlas_v2.experiments import summarize
for dataset_name in PUBLICATION_CONFIG["protocols"]:
    for protocol_name in ("natural", "minority_challenge"):
        root = RESULTS/dataset_name/"publication"/protocol_name
        print(dataset_name, protocol_name)
        display(summarize(root))
'''

DIAGNOSTICS = '''from atlas_v2.experiments import diagnostics, summarize
main_runs = []
for dataset_name, dataset_protocols in PUBLICATION_CONFIG["protocols"].items():
    for protocol_name, spec in dataset_protocols.items():
        root = RESULTS/dataset_name/"publication"/protocol_name
        candidates = []
        for complete in root.glob("atlas_*/complete.json"):
            payload = json.loads(complete.read_text())
            if payload["config"]["seed"] == 42 and payload["config"]["loss"] == PUBLICATION_CONFIG["loss"]:
                candidates.append(complete.parent)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one seed-42 main run for {dataset_name}/{protocol_name}; found {len(candidates)}")
        run_dir = candidates[0]
        diagnostics(spec["prepared"], run_dir, samples=256, repeats=100)
        main_runs.append(run_dir)
        summarize(root)
print("Diagnosed main runs:", *main_runs, sep="\\n")
'''

AGENT_AUDIT = '''from atlas_v2.agent import evaluate_agent
agent_results = evaluate_agent(main_runs, RESULTS/"agent_evaluation")
display(agent_results)
if not agent_results["complete"].all():
    raise RuntimeError("At least one real-run agent evaluation is incomplete.")
if not (agent_results["unsupported_claim_rate"] == 0).all():
    raise RuntimeError("The evidence agent emitted unsupported claims.")
'''

FINAL_AUDIT = '''rows = []
for dataset_name in PUBLICATION_CONFIG["protocols"]:
    target = PUBLICATION_CONFIG["submitted_reference"][dataset_name]
    for protocol_name in ("natural", "minority_challenge"):
        all_runs = pd.read_csv(RESULTS/dataset_name/"publication"/protocol_name/"all_runs.csv")
        main = all_runs[(all_runs.variant == "atlas") & (all_runs.loss == PUBLICATION_CONFIG["loss"])]
        rows.append({
            "dataset": dataset_name, "protocol": protocol_name, "completed_main_seeds": main.seed.nunique(),
            "accuracy_mean": main.accuracy.mean(), "weighted_f1_mean": main.weighted_f1.mean(),
            "macro_f1_mean": main.macro_f1.mean(), "mcc_mean": main.mcc.mean(),
            "submitted_accuracy_target": target["accuracy"],
            "submitted_weighted_f1_target": target["weighted_f1"],
            "accuracy_above_submitted_target": bool(main.accuracy.mean() > target["accuracy"]),
            "weighted_f1_above_submitted_target": bool(main.weighted_f1.mean() > target["weighted_f1"]),
        })
audit = pd.DataFrame(rows)
audit.to_csv(RESULTS/"final_implementation_audit.csv", index=False)
display(audit)
print("Target comparisons are descriptive. They do not permit test-set tuning or result manipulation.")
'''


def write_notebook(name, title, cells):
    notebook_cells = [{
        'cell_type': 'markdown',
        'metadata': {},
        'source': [f'# {title}\n', 'Run cells in order. Durable artifacts are written to the configured Drive results folder.\n'],
    }]
    notebook_cells.extend({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': cell.splitlines(True),
    } for cell in cells)
    payload = {
        'cells': notebook_cells,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'nbformat': 4,
        'nbformat_minor': 4,
    }
    path = ROOT / 'notebooks' / f'{name}.ipynb'
    path.write_text(json.dumps(payload, indent=1))


write_notebook(
    '04_prepare_publication_protocols',
    'Natural-prior and minority-coverage preparation',
    [BOOTSTRAP, SETUP, CONFIGURE_PROTOCOLS, PREPARE_PROTOCOLS, AUDIT_PREPARATION],
)
write_notebook(
    '05_publication_controlled_experiments',
    'Repeated-seed baselines, legacy comparison and matched ablations',
    [BOOTSTRAP, SETUP, LOAD, BUILD_PLAN, RUN_EXPERIMENTS, SUMMARIZE],
)
write_notebook(
    '06_diagnostics_and_evidence_agent',
    'Latency, explanation controls, evidence agent and implementation audit',
    [BOOTSTRAP, SETUP, LOAD, DIAGNOSTICS, AGENT_AUDIT, FINAL_AUDIT],
)
