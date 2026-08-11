# Reproducing the study

Every number in the manuscript comes from an artefact in `results/`. This file says
which command produces which artefact, and what to check if a number does not match.

## The chain

```
00_download_data.py       Kaggle corpus -> data/raw/          verifies 780/374/793
        |
01_preprocess_and_split.py
        |-- intensity_analysis.json     bone-intensity means, SDs, Kruskal-Wallis
        |-- data/splits/manifest.json   the partition, asserted leakage-free
        `-- data/processed/             320x320 CLAHE PNGs + balanced train index
        |
02_train.py               -> checkpoints/full/checkpoint.pt
        `-- results/run_full.json       history, best epoch, full test report
        `-- results/predictions_full.npz  probabilities + labels, for everything below
        |
        +-- 03_ablation.py       seven variants          -> ablation.json
        +-- 04_baselines.py      four backbones + probe  -> baselines.json
        +-- 05_multiseed.py      three seeds             -> multiseed.json
        +-- 06_efficiency.py     params/MACs/latency     -> efficiency.json
        +-- 07_fusion_sweep.py   w sensitivity           -> fusion_sweep.json
        +-- 08_explainability.py CAM/GNN/LIME + deletion -> explainability.json
        |
09_make_figures.py        every measured figure -> figures/ (and the repo root)
10_export_paper_values.py results.json + paper_values.tex (the LaTeX macros)
11_compare_to_paper.py    diffs this run against results/paper_reported.json
```

Two ablation rows — (ii) *w/o auxiliary branch* and (vi) *w/o test-time augmentation* —
are inference-time changes to the retained checkpoint. They consume no training budget
and involve no new weights, so those rows are exact rather than approximate.

## Running it

### On a GPU you control

```bash
pip install -e ".[xai,dev]"
python scripts/00_download_data.py
python scripts/01_preprocess_and_split.py
python scripts/02_train.py
python scripts/03_ablation.py && python scripts/04_baselines.py
python scripts/05_multiseed.py && python scripts/06_efficiency.py
python scripts/07_fusion_sweep.py && python scripts/08_explainability.py
python scripts/09_make_figures.py && python scripts/10_export_paper_values.py
```

### On Kaggle (what actually produced these results)

`tools/kaggle_runner.py` packs `src/`, `configs/` and `scripts/` into a base64 tarball
embedded in the kernel script, so the pushed script is a complete copy of the code that
produced the numbers.

```bash
export KAGGLE_API_TOKEN=KGAT_...            # or ~/.kaggle/access_token
python tools/kaggle_runner.py push --stage stage1_main
# then, once, in the notebook UI: Settings -> Accelerator -> GPU T4 x2 -> Save Version
python tools/kaggle_runner.py wait  --stage stage1_main
python tools/kaggle_runner.py fetch --stage stage1_main
```

Repeat for `stage2_ablation`, `stage3_baselines`, `stage4_multiseed`. Later stages
attach earlier ones as kernel data sources, so the full model's checkpoint carries
forward without retraining.

**Two manual steps, and why they cannot be automated.** The push API accepts the
relevant fields and then ignores them, so both must be set in the notebook UI under
**Settings**, and **an API push resets both**. Push first, then set them, then Save
Version — in that order, or the push undoes the settings.

1. **Accelerator → GPU T4 x2.** The default is a **P100 (sm_60)**, which the current
   Kaggle PyTorch build does not support; CUDA raises `no kernel image is available for
   execution on the device` on the first launch. The push body's `enableGpu` is only a
   boolean, and passing `machineShape: "GPU_T4X2"` returns HTTP 200 and still allocates
   a P100. The entrypoint refuses to run on a pre-Volta device rather than silently
   producing nothing:

   ```python
   if torch.cuda.get_device_capability(0)[0] < 7:
       raise SystemExit("... set the accelerator to 'GPU T4 x2' in the notebook UI")
   ```

2. **Internet → on.** The push body sets `enableInternet: True` and this is likewise not
   honoured. Without it the torchvision ImageNet weights cannot be downloaded and
   training dies partway through the first run with

   ```
   urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
   ```

   The Settings menu item is a toggle whose label states the action, not the state:
   "Turn on internet" means internet is currently **off**.

Verify both before Save Version. A wrong accelerator costs 60 seconds; missing internet
costs about six minutes, because preprocessing completes first.

Note also that the mount path for an attached dataset varies between
`/kaggle/input/<slug>/...` and `/kaggle/input/datasets/<owner>/<slug>/...`. The entrypoint
resolves it by looking for the directory that holds the three class folders rather than
assuming either layout, and prints what it resolved to.

## Checking a run

```bash
pytest tests/ -q                       # protocol and graph-layer invariants
python scripts/11_compare_to_paper.py  # diff against results/paper_reported.json
```

`11_compare_to_paper.py` exits non-zero when a headline metric moves more than the
tolerance. That is information, not a bug: it means the run did not land on the same
numbers, and the manuscript reports what was measured here.

## What is and is not deterministic

Seeded: NumPy, Python `random`, torch CPU and CUDA RNGs, the DataLoader generator and
its worker seeds, the offline augmentation plan, and the partition. cuDNN is pinned to
deterministic algorithms (`utils.set_seed`), because convolution autotuning is the
largest source of run-to-run drift on identical inputs.

Not fully pinned: mixed-precision accumulation order on GPU, and any change of PyTorch,
cuDNN or GPU model. Expect small movement across hardware; the multi-seed table is the
honest statement of how much movement to expect across runs.

Every `results/run_*.json` records the environment it was produced in — Python, torch,
CUDA, GPU name and capability, and library versions — so a divergence can be attributed
rather than guessed at.
