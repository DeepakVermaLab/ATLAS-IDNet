"""Regenerate runnable notebooks and a self-contained Colab source bundle."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
bootstrap='''from pathlib import Path
import os, sys, json, errno, time
# VS Code + Colab: load the synchronized source folder, without browser uploads.
DRIVE = Path("/content/drive/MyDrive")
try:
    from google.colab import drive
except ImportError:
    drive = None
def with_drive_retry(operation, attempts=3, reconnect=None):
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as exc:
            disconnected = exc.errno in {errno.ENOTCONN, errno.EIO, errno.ESTALE} or "Transport endpoint" in str(exc)
            if not disconnected or (drive is None and reconnect is None) or attempt == attempts-1:
                raise
            print(f"Drive disconnected. Reconnecting and retrying ({attempt+1}/{attempts-1})...", flush=True)
            if reconnect is not None:
                reconnect()
            else:
                drive.mount("/content/drive", force_remount=True)
            time.sleep(1)

def ensure_drive():
    if drive is not None:
        if not DRIVE.is_dir():
            drive.mount("/content/drive", force_remount=False)
        # Probe the filesystem: an existing mount directory can still be disconnected.
        next(DRIVE.iterdir(), None)
with_drive_retry(ensure_drive)

def resolve_v2(bases):
    checked = []
    for base in bases:
        for candidate in (base, base/"implementation/v2", base/"v2"):
            checked.append(str(candidate))
            required = ["src/atlas_v2/__init__.py", "src/atlas_v2/data.py",
                        "src/atlas_v2/models.py", "src/atlas_v2/experiments.py", "requirements.txt"]
            if all((candidate/name).is_file() for name in required):
                return candidate.resolve()
    raise FileNotFoundError(
        "The v2 Python source is not visible to this kernel. "
        "Expected the synchronized implementation/v2 folder on mounted Drive. "
        "Set ATLAS_V2_ROOT to that folder if your Drive location differs. Checked: "
        + "; ".join(checked))

manual = os.environ.get("ATLAS_V2_ROOT")
bases = [Path(manual).expanduser()] if manual else []
bases += [DRIVE/"projects/implementation/ATLAS-IDNet/implementation/v2",
          DRIVE/"projects/ATLAS-IDNet", DRIVE/"ATLAS-IDNet",
          Path.cwd(), *Path.cwd().parents, Path("/content/ATLAS-IDNet")]
V2 = with_drive_retry(lambda: resolve_v2(bases))
sys.path.insert(0, str(V2/"src"))
for name in list(sys.modules):
    if name == "atlas_v2" or name.startswith("atlas_v2."):
        del sys.modules[name]
import atlas_v2
assert Path(atlas_v2.__file__).resolve().parent == (V2/"src/atlas_v2").resolve()
print("Code:", V2)
print("Source import OK. No file upload or model weights required.")
'''
setup='''import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(V2/"requirements.txt")])
try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None:
    drive.mount("/content/drive")
import torch, pandas, sklearn
print("CPU runtime:", torch.__version__, "threads:", os.cpu_count())
'''
config='''# Review these paths before running. Results go to a NEW v2 directory.
DRIVE = Path("/content/drive/MyDrive")
BASE = DRIVE/"projects/Dataset/Security" if DRIVE.exists() else Path("/Volumes/LocalDrive1/Dataset/Security")
RESULTS = DRIVE/"projects/implementation/ATLAS-IDNet/results/v2" if DRIVE.exists() else V2/"results"
RESULTS.mkdir(parents=True, exist_ok=True)
CONFIG = {
    "results": str(RESULTS), "seeds": [42, 43, 44, 45, 46], "epochs": 30,
    "datasets": {
        "CICIoT2023": {"csv_globs": [str(BASE/"CICIoT2023/**/*.csv")], "label": "label"},
        "NF-UQ-NIDS-v2": {"csv_globs": [str(BASE/"NF-UQ-NIDS-v2/**/*.csv")], "label": "Attack"}
    }
}
for spec in CONFIG["datasets"].values():
    spec.update(max_rows=100000, split="stratified", group_column=None, time_column=None,
        exclude_columns=["Label", "label", "Attack", "Attack_type", "Class", "Dataset", "dataset",
                         "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Flow ID", "Src IP", "Dst IP", "Timestamp"])
CONFIG_PATH = RESULTS/"experiment_config.json"
if CONFIG_PATH.exists():
    CONFIG = json.loads(CONFIG_PATH.read_text())
    print("Reused existing config; edit explicitly to change paths/protocols.")
else:
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2))
print(json.dumps(CONFIG, indent=2))
os.environ["ATLAS_V2_CONFIG"] = str(CONFIG_PATH)
'''
load='''# Set this explicitly if your results directory differs.
default = "/content/drive/MyDrive/projects/implementation/ATLAS-IDNet/results/v2/experiment_config.json"
CONFIG_PATH = Path(os.environ.get("ATLAS_V2_CONFIG", default if Path("/content").exists() else str(V2/"results/experiment_config.json")))
CONFIG = json.loads(CONFIG_PATH.read_text())
RESULTS = Path(CONFIG["results"])
'''
prepare='''from atlas_v2.data import prepare, PREPARATION_VERSION
import hashlib
for name, spec in CONFIG["datasets"].items():
    spec = {k: v for k, v in spec.items() if k != "prepared"}
    key = hashlib.sha256(json.dumps({"spec": spec, "preparation_version": PREPARATION_VERSION}, sort_keys=True).encode()).hexdigest()[:12]
    folder = RESULTS/name/("prepared_"+key)
    # Existing prepared data is immutable. Change the spec to create a distinct preparation.
    def prepare_one():
        if not (folder/"metadata.json").exists():
            prepare(spec, folder)
        print(name, json.loads((folder/"metadata.json").read_text())["counts"])
        CONFIG["datasets"][name]["prepared"] = str(folder)
        temp = CONFIG_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(CONFIG, indent=2))
        os.replace(temp, CONFIG_PATH)
    with_drive_retry(prepare_one)
'''
pilot='''from atlas_v2.experiments import run
for name, spec in CONFIG["datasets"].items():
    path = run(spec["prepared"], RESULTS/name/"pilot", epochs=2, seed=42)
    print(name, json.loads((path/"metrics.json").read_text()))
print("Pilot only: not final paper results. Review duration before notebook 02.")
'''
train='''from atlas_v2.experiments import run
VARIANTS = ["atlas", "mlp", "single_residual", "no_residual", "mean", "concat", "random_groups",
            "random_forest", "extra_trees", "hist_gradient", "logistic"]
for name, spec in CONFIG["datasets"].items():
    meta = json.loads((Path(spec["prepared"])/"metadata.json").read_text())
    if not meta.get("full_class_evaluation_supported", False):
        raise RuntimeError(f"{name}: insufficient class coverage. Review missing_classes in metadata and increase max_rows before full experiments.")
    for variant in VARIANTS:
        for seed in CONFIG["seeds"]:
            run(spec["prepared"], RESULTS/name/"experiments", variant=variant, seed=seed, epochs=CONFIG["epochs"])
    for loss in ["balanced", "focal"]:
        for seed in CONFIG["seeds"]:
            run(spec["prepared"], RESULTS/name/"experiments", seed=seed, loss=loss, epochs=CONFIG["epochs"])
'''
diag='''from atlas_v2.experiments import diagnostics, summarize
for name, spec in CONFIG["datasets"].items():
    root = RESULTS/name/"experiments"
    for done in sorted(root.glob("*/complete.json")):
        diagnostics(spec["prepared"], done.parent)
    display(summarize(root))
'''
for name,title,codes in [
    ('00_setup_and_paths','Environment and persistent paths',[bootstrap,setup,config]),
    ('01_prepare_and_cpu_pilot','Train-only preprocessing and CPU feasibility pilot',[bootstrap,setup,load,prepare,pilot]),
    ('02_controlled_experiments','CPU baselines, architectural controls and loss comparisons',[bootstrap,setup,load,train]),
    ('03_latency_explanations_summary','CPU latency, occlusion faithfulness and seed statistics',[bootstrap,setup,load,diag])]:
    cells=[{'cell_type':'markdown','metadata':{},'source':[f'# {title}\n', 'Run cells in order. Notebook outputs remain visible; durable artifacts are written to results/v2.\n']}]
    cells += [{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':c.splitlines(True)} for c in codes]
    path=ROOT/'notebooks'/f'{name}.ipynb'; path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':4},indent=1))
with ZipFile(ROOT/'atlas_v2_colab.zip','w',ZIP_DEFLATED) as z:
    for folder in ['src','notebooks']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._'):
                z.write(p,p.relative_to(ROOT))
    z.write(ROOT/'requirements.txt','requirements.txt')
