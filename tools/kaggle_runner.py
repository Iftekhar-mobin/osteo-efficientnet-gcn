"""Drive Kaggle GPU kernels over the REST API.

The whole repository (src/, configs/, scripts/) is packed into the kernel script itself
as a base64 tarball, so a run is reproducible from one artefact: the pushed script is a
complete, self-contained copy of the code that produced the numbers.

Authentication uses a Kaggle API token read from ~/.kaggle/access_token (or the
KAGGLE_API_TOKEN environment variable). The token is never written into a kernel.

Note on accelerators: the push API exposes only a boolean "GPU on", and Kaggle's default
GPU is a P100 whose sm_60 compute capability the current Kaggle PyTorch build does not
support. The accelerator must therefore be set to "GPU T4 x2" once per kernel in the
notebook UI; ``--check`` reports which device a finished run actually used.

    python tools/kaggle_runner.py push  --stage stage1_main
    python tools/kaggle_runner.py wait  --stage stage1_main
    python tools/kaggle_runner.py fetch --stage stage1_main
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.kaggle.com/api/v1"
USER = os.environ.get("KAGGLE_USERNAME", "iftekharulmobin")
DATASET = "mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset"
PACK = ("src", "configs", "scripts")


def token() -> str:
    value = os.environ.get("KAGGLE_API_TOKEN")
    if value:
        return value.strip()
    path = Path.home() / ".kaggle" / "access_token"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("no Kaggle token: set KAGGLE_API_TOKEN or ~/.kaggle/access_token")


def headers() -> dict:
    return {"Authorization": "Bearer " + token()}


def pack_repo() -> str:
    """base64(tar.gz) of the code directories -- no data, no checkpoints, no secrets."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in PACK:
            root = REPO_ROOT / name
            for path in sorted(root.rglob("*")):
                if path.is_dir() or "__pycache__" in path.parts:
                    continue
                if path.suffix in {".pyc", ".pt", ".pth"}:
                    continue
                tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------------------------
# Stages. Each is a list of shell-ish command argument lists run in order.
# ---------------------------------------------------------------------------------
RAW = ("/kaggle/input/datasets/mohamedgobara/"
       "multi-class-knee-osteoporosis-x-ray-dataset/OS Collected Data")

STAGES: dict[str, dict] = {
    "stage1_main": {
        "title": "Osteo Stage1 Main",
        "needs": [],
        "steps": [
            ["scripts/01_preprocess_and_split.py", "--raw-dir", RAW],
            ["scripts/02_train.py", "--set", "run_name=full"],
            ["scripts/06_efficiency.py", "--run", "full"],
            ["scripts/07_fusion_sweep.py", "--run", "full"],
            ["scripts/08_explainability.py", "--run", "full"],
            ["scripts/09_make_figures.py", "--run", "full"],
        ],
    },
    "stage2_ablation": {
        "title": "Osteo Stage2 Ablation",
        "needs": ["stage1_main"],
        "steps": [
            ["scripts/01_preprocess_and_split.py", "--raw-dir", RAW, "--skip-intensity"],
            ["scripts/01_preprocess_and_split.py", "--raw-dir", RAW, "--skip-intensity",
             "--set", "preprocess.clahe=false"],
            ["scripts/01_preprocess_and_split.py", "--raw-dir", RAW, "--skip-intensity",
             "--set", "augment.balance=false",
             "--set", "data.processed_dir=data/processed_unbalanced"],
            ["scripts/03_ablation.py"],
            ["scripts/09_make_figures.py", "--run", "full"],
        ],
    },
    "stage3_baselines": {
        "title": "Osteo Stage3 Baselines",
        "needs": ["stage1_main"],
        "steps": [
            ["scripts/01_preprocess_and_split.py", "--raw-dir", RAW, "--skip-intensity"],
            ["scripts/04_baselines.py"],
        ],
    },
    "stage4_multiseed": {
        "title": "Osteo Stage4 Multiseed",
        "needs": ["stage1_main"],
        "steps": [
            ["scripts/05_multiseed.py", "--seeds", "42", "7", "1234",
             "--raw-dir", RAW, "--skip-existing"],
            ["scripts/09_make_figures.py", "--run", "full"],
        ],
    },
}


ENTRYPOINT = r'''
import base64, io, os, subprocess, sys, tarfile, shutil, glob, time, json
from pathlib import Path

WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
REPO = Path("/kaggle/repo"); REPO.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)

# 1. unpack the code shipped inside this script
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
    tar.extractall(REPO)
print("unpacked:", sorted(p.name for p in REPO.iterdir()), flush=True)

# 2. carry forward artefacts from earlier stages (attached kernel outputs).
# Kaggle nests attached inputs (datasets under /kaggle/input/datasets/<owner>/...,
# notebook outputs elsewhere), so search rather than assume a fixed depth.
carried = []
for wanted in ("results", "checkpoints"):
    for src in sorted(Path("/kaggle/input").rglob(wanted)):
        if not src.is_dir() or "OS Collected Data" in str(src):
            continue
        dest = REPO / wanted
        dest.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if item.is_file():
                target = dest / item.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(item, target)
        carried.append(str(src))
print("carried forward:", carried or "nothing", flush=True)
if not carried and NEEDS_PRIOR:
    print("input tree:", flush=True)
    for p in sorted(Path("/kaggle/input").glob("*/*/*"))[:40]:
        print("   ", p, flush=True)
    raise SystemExit("this stage needs an earlier stage's output but none was "
                     "attached -- check the notebook's Input panel")

# 3. dependencies missing from the Kaggle image
for pkg in EXTRA_PIP:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        raise SystemExit(f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} is "
                         f"not supported by this PyTorch build -- set the accelerator "
                         f"to 'GPU T4 x2' in the notebook UI and re-run")

# 4. run the stage
failures = []
for step in STEPS:
    print("\n" + "=" * 78 + f"\n$ python {' '.join(step)}\n" + "=" * 78, flush=True)
    started = time.time()
    proc = subprocess.run([sys.executable] + step)
    print(f"-- exit {proc.returncode} in {time.time()-started:.0f}s", flush=True)
    if proc.returncode != 0:
        failures.append(step)

# 5. publish artefacts (results, figures, checkpoints) to the kernel output
for name in ("results", "figures", "checkpoints", "data/splits"):
    src = REPO / name
    if not src.exists():
        continue
    dest = WORK / Path(name).name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
for png in REPO.glob("*.png"):
    shutil.copy2(png, WORK / png.name)

json.dump({"stage": STAGE, "failures": [" ".join(f) for f in failures]},
          open(WORK / "stage_status.json", "w"), indent=2)
print("\nDONE", STAGE, "failures:", failures, flush=True)
if failures:
    raise SystemExit(1)
'''


def build_script(stage: str) -> str:
    spec = STAGES[stage]
    payload = pack_repo()
    header = (
        f"PAYLOAD = {payload!r}\n"
        f"STAGE = {stage!r}\n"
        f"STEPS = {json.dumps(spec['steps'])}\n"
        f"EXTRA_PIP = {json.dumps(['lime'])}\n"
        f"NEEDS_PRIOR = {bool(spec['needs'])}\n"
    )
    return header + ENTRYPOINT


def slug(stage: str) -> str:
    return "osteo-" + stage.replace("_", "-")


def push(stage: str) -> dict:
    spec = STAGES[stage]
    body = {
        "id": None,
        "slug": f"{USER}/{slug(stage)}",
        "newTitle": spec["title"],
        "text": build_script(stage),
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "enableTpu": False,
        "enableInternet": True,
        "datasetDataSources": [DATASET],
        "competitionDataSources": [],
        "kernelDataSources": [f"{USER}/{slug(n)}" for n in spec["needs"]],
        "modelDataSources": [],
        "categoryIds": [],
        "docker": "latest",
    }
    response = requests.post(BASE + "/kernels/push", headers=headers(), json=body,
                             timeout=600)
    response.raise_for_status()
    return response.json()


def status(stage: str) -> dict:
    response = requests.get(BASE + "/kernels/status", headers=headers(),
                            params={"userName": USER, "kernelSlug": slug(stage)},
                            timeout=120)
    return response.json()


def wait(stage: str, poll: int = 60, limit: int = 43200) -> dict:
    started, last = time.time(), None
    while time.time() - started < limit:
        current = status(stage)
        state = current.get("status")
        if state != last:
            print(f"[{int(time.time()-started):6d}s] {stage}: {state} "
                  f"{current.get('failureMessage') or ''}", flush=True)
            last = state
        if state in {"complete", "error", "cancelAcknowledged"}:
            return current
        time.sleep(poll)
    return {"status": "timeout"}


def output(stage: str) -> dict:
    response = requests.get(BASE + "/kernels/output", headers=headers(),
                            params={"userName": USER, "kernelSlug": slug(stage)},
                            timeout=600)
    response.raise_for_status()
    return response.json()


def print_log(stage: str, tail: int = 120) -> None:
    raw = output(stage).get("log", "")
    try:
        events = json.loads(raw)
        lines = [e["data"] for e in events]
    except Exception:
        lines = [raw]
    text = "".join(lines).splitlines()
    for line in text[-tail:]:
        print(line.encode("ascii", "replace").decode())


def fetch(stage: str, dest: Path = REPO_ROOT) -> list[str]:
    """Download the kernel's output files into the repository."""
    data = output(stage)
    written = []
    for item in data.get("files", []):
        url, name = item.get("url"), item.get("fileName")
        if not url:
            continue
        response = requests.get(url, headers=headers(), timeout=1200)
        response.raise_for_status()
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        written.append(str(target))
    # Kaggle bundles multi-file output as output.zip on some endpoints.
    for path in list(written):
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(dest)
            written.append(f"{path} (extracted)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["push", "status", "wait", "log", "fetch",
                                           "run", "stages"])
    parser.add_argument("--stage", default=None)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--tail", type=int, default=120)
    args = parser.parse_args()

    if args.action == "stages":
        for name, spec in STAGES.items():
            print(f"{name:18s} needs={spec['needs']}  {len(spec['steps'])} steps")
        return 0
    if not args.stage or args.stage not in STAGES:
        raise SystemExit(f"--stage must be one of {list(STAGES)}")

    if args.action == "push":
        print(json.dumps(push(args.stage), indent=2))
        print(f"\nNOW: open https://www.kaggle.com/code/{USER}/{slug(args.stage)}/edit,"
              f"\n     set ACCELERATOR to 'GPU T4 x2', then Save Version "
              f"(Save & Run All).")
    elif args.action == "status":
        print(json.dumps(status(args.stage), indent=2))
    elif args.action == "wait":
        print(json.dumps(wait(args.stage, args.poll), indent=2))
    elif args.action == "log":
        print_log(args.stage, args.tail)
    elif args.action == "fetch":
        for path in fetch(args.stage):
            print(path)
    elif args.action == "run":
        push(args.stage)
        wait(args.stage, args.poll)
        print_log(args.stage, args.tail)
        for path in fetch(args.stage):
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
