# EfficientNet-GCN

**A Hybrid Convolutional–Graph Framework for Explainable Multi-Class Osteoporosis
Severity Classification from Knee Radiographs**

Reference implementation and complete experimental record for the manuscript. An
ImageNet-pretrained EfficientNetB0 trunk produces a shared feature map; that map is
channel-reduced and reformulated as a 100-node, eight-connected **patch graph**; three
residual graph-convolutional layers propagate relational context over it and read it out
through concatenated mean–max pooling; a graph-free auxiliary branch runs in parallel;
and the two are fused by a weighted probability head.

Tasnim Sultana Sintheia, Musfique Rahman Muin, Pranta Das Gupta, and Md. Iftekharul
Mobin. *EfficientNet-GCN: A Hybrid Convolutional–Graph Framework for Explainable
Multi-Class Osteoporosis Severity Classification from Knee Radiographs.*

---

## The idea in one paragraph

Every CNN approach to this task reasons over the radiograph as a Euclidean grid:
anatomically adjacent bone regions influence the prediction only implicitly, through the
receptive field of a convolution, never through an explicit relational representation.
EfficientNet-GCN makes that relation explicit. The backbone's terminal
$1280\times10\times10$ map is treated as one hundred patches; each patch becomes a node
linked to its orthogonal and diagonal neighbours; and three graph-convolutional layers
propagate over that structure before a dual readout summarises both the joint-wide
density signal (mean pooling) and the single most abnormal patch (max pooling). The
second contribution is protocol rather than architecture: the corpus is partitioned
**before** any augmentation and only the training partition is rebalanced, so no
augmented variant of a test radiograph can appear in training — a source of optimistic
bias that is widely present in this literature.

```
                   ┌──── shared backbone ────┐┌── relational ──┐┌── fusion ──┐

                                          ┌─▶ Proj 1×1 ─▶ patch graph ─▶ GCN×3 ─▶ mean‖max ─▶ MLP ─┐
  radiograph ─▶ CLAHE ─▶ 320² ─▶ EffNetB0 ┤   1280→256    100 nodes      residual    512-d        ├─▶ 0.6/0.4 ─▶ P
                                          └─▶ GAP ────────────────────────────────────────▶ MLP ─┘
                       ▲                        auxiliary branch: graph-free path to logits
                       │
              partition happens HERE, before augmentation
```

| Component | What it does | Parameters |
|---|---|---|
| **EfficientNetB0 encoder** | last 6 of 9 top-level blocks trainable; stem + first two MBConv stages frozen at ImageNet weights | 4.01 M |
| **Patch-Graph Construction** | 1×1 conv 1280→256, then 100 nodes on an 8-connected lattice (684 directed edges), self-looped and symmetrically normalised | 0.33 M |
| **Residual Dual-Pooling Graph Branch** | 3 × (256→256) graph convolutions, residuals around layers 2–3, mean‖max readout → 512-d | 0.36 M |
| **Auxiliary Branch + Weighted Ensemble Head** | GAP over the *un-reduced* map → 1280→256→3; fused $0.6\,\mathrm{softmax}(z_{\mathrm{GCN}}) + 0.4\,\mathrm{softmax}(z_{\mathrm{CNN}})$ | 0.33 M |
| **Total** | | **5,027,970** |

The projection is not decoration. A graph layer consuming the raw 1,280-channel features
would need a $1280\times1280$ weight matrix — about 1.64 M parameters in a *single*
layer. The 1×1 convolution performs the reduction with 0.33 M and leaves each graph
layer at $256\times256$, or 65.5 K.

---

## Quick start

```bash
git clone <this repository> && cd osteo-efficientnet-gcn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[xai,dev]"

python scripts/00_download_data.py          # public Kaggle corpus, no credentials needed
python scripts/01_preprocess_and_split.py   # CLAHE, split, balance -- ~4 min
python scripts/02_train.py                  # the proposed model
python scripts/09_make_figures.py
```

Everything lands in `results/` (CSV + JSON + one `results.json`) and `figures/` (PNG at
300 dpi). `results/paper_values.tex` is a LaTeX macro file the manuscript reads its
numbers from, so a value in the text cannot drift away from the run that produced it.

A GPU is effectively required: at $320\times320$ the backbone runs about 10 images/s on
8 CPU cores, which puts one 45-epoch run near four hours. See
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the Kaggle path that produced these
results.

---

## Results

<!--RESULTS-->

---

## Leakage discipline

The single most important property of this codebase, and the manuscript's main
methodological claim:

1. **The split happens before any augmentation.** `scripts/01` partitions the
   preprocessed originals 70/15/15 per class, and only then manufactures augmented
   variants — for the training partition alone.
2. **The check compares source stems, not paths.** After augmentation a file is named
   `<stem>_aug<k>.png`; comparing full paths would pass trivially while a variant of a
   test radiograph sat in training. `verify_no_leakage` strips `_aug<k>` first, and
   `tests/test_protocol_integrity.py` asserts that a planted violation is caught.
3. **The check runs twice** — immediately after the split, and again on the
   materialised files after augmentation.
4. **Validation and test keep their natural class distribution.** Only training is
   rebalanced to 1,000 per class. Every metric is therefore reported as a macro average.
5. **Validation and test see no stochastic transform at all** — resize plus ImageNet
   normalisation, nothing else.
6. **The fusion weight is swept on validation, never on test.** `scripts/07` selects $w$
   on the validation partition and reports test numbers at both the a-priori and the
   selected value.
7. **The partition manifest is committed.** `data/splits/manifest.json` lists every
   source filename and its partition, so the split is auditable without redistributing
   a single image.

**What this does *not* establish.** The source dataset carries no patient identifiers.
The partition is verified free of *image*-level overlap; if the corpus contains
bilateral pairs or repeat studies from one individual, *patient*-level leakage cannot be
excluded and the reported metrics would be optimistic. This limitation applies to every
published study using this corpus.

---

## Repository layout

```
.
├── configs/default.yaml           every value maps to a row of the hyperparameter table
├── data/
│   ├── README.md                  where to get the corpus; what it does not document
│   └── splits/manifest.json       the committed, auditable partition
├── src/osteognn/
│   ├── config.py                  typed config tree; ABLATIONS / BASELINES / probe
│   ├── train.py                   warmup+cosine, differential LRs, AMP, early stopping
│   ├── metrics.py                 every reported quantity, incl. QWK, ECE, Wilson, McNemar
│   ├── efficiency.py              parameters, MACs, latency, peak memory
│   ├── figures.py                 every measured figure, one validated palette
│   ├── utils.py                   seeding, device/environment report
│   ├── data/                      preprocess.py · split.py · augment.py · datasets.py
│   ├── models/                    patch_graph.py · gcn_branch.py · encoder.py · ensemble.py
│   └── xai/                       gradcam.py · gnn_explainer.py · lime_explain.py · faithfulness.py
├── scripts/                       00–11, numbered in execution order
├── tests/                         protocol integrity + GCN equivalence
├── tools/kaggle_runner.py         packs the repo into a Kaggle GPU kernel
├── results/                       CSV + JSON + results.json + paper_values.tex
├── figures/                       regenerated PNGs at 300 dpi
├── docs/
│   ├── REPRODUCING.md             the artefact chain, end to end
│   └── legacy_figures/            the pre-reproduction figures, kept for provenance
└── paper.tex                      the manuscript, reading numbers from results/
```

---

## Scripts

| Script | Purpose | Produces |
|---|---|---|
| `00_download_data.py` | Fetch and verify the Kaggle corpus | `data/raw/`, class-count check |
| `01_preprocess_and_split.py` | CLAHE at native resolution, split, balance | `intensity_analysis.json`, `manifest.json`, `data/processed/` |
| `02_train.py` | Train one configuration, evaluate on test | `run_<name>.json`, `predictions_<name>.npz` |
| `03_ablation.py` | The seven-variant protocol | `ablation.json`, `table_ablation.csv` |
| `04_baselines.py` | Four backbones under the identical protocol, + ordering probe | `baselines.json`, `table_baselines.csv` |
| `05_multiseed.py` | Repeat across seeds; the variance a single run cannot show | `multiseed.json` |
| `06_efficiency.py` | Parameters, MACs, latency, peak memory | `efficiency.json` |
| `07_fusion_sweep.py` | Sensitivity to the ensemble weight $w$ | `fusion_sweep.json` |
| `08_explainability.py` | Grad-CAM, GNNExplainer, LIME + deletion/insertion | `explainability.json`, XAI figures |
| `09_make_figures.py` | Redraw every measured figure | `figures/*.png` |
| `10_export_paper_values.py` | Aggregate everything; emit the LaTeX macros | `results.json`, `paper_values.tex` |
| `11_compare_to_paper.py` | Diff against `paper_reported.json`, non-zero on breach | `compare_to_paper.csv` |

Ablations and baselines are declarative — each is a set of dotted overrides on the
default config, in `src/osteognn/config.py`, and a test asserts that every ablation
changes **exactly one** config section. That is what makes the table a controlled
comparison rather than a collection of separately tuned runs.

```python
ABLATIONS = {
    "no_gcn":       {"model.use_gcn_branch": False},        # (i)
    "no_aux":       {"inference.fusion_weight": 1.0},       # (ii)  inference-only
    "mean_pool":    {"model.gcn.pooling": "mean"},          # (iii)
    "no_clahe":     {"preprocess.clahe": False},            # (iv)
    "no_balance":   {"augment.balance": False},             # (v)
    "no_tta":       {"inference.tta_hflip": False},         # (vi)  inference-only
    "conv_control": {"model.conv_control": True},           # (vii)
}
```

---

## The graph layer, and why it is dense

The manuscript states the propagation as
$H^{(l+1)} = \phi(\hat{A} H^{(l)} W^{(l)})$. At 100 nodes the dense form is a single
batched matmul — faster than a sparse scatter, deterministic, and dependency-free. That
is only an implementation detail if the two agree numerically, so
`tests/test_gcn_equivalence.py` asserts it against PyTorch Geometric's `GCNConv`, the
reference implementation the citation points at:

```python
conv = GCNConv(32, 32, add_self_loops=True, normalize=True).eval()
reference = conv(x, edge_index)
dense     = a_hat @ x @ conv.lin.weight.T + conv.bias
assert torch.allclose(reference, dense, atol=1e-5)
```

The same file checks the edge count the manuscript reports — 684 directed edges over a
10×10 lattice: 4 corners of degree 3, 32 border patches of degree 5, 64 interior patches
of degree 8 — and that the adjacency is symmetric with a degree-dependent diagonal.

**The objection this design has to answer.** Because $A$ is a fixed eight-connected
lattice, identical for every input image, the propagation computes a degree-normalised
sum over each node's 3×3 neighbourhood followed by a shared linear map — structurally
close to a 3×3 convolution. Variant (vii) is the experiment that settles whether the
graph does anything a convolution cannot: it substitutes a **parameter-matched** 3×3
conv stack operating on the same projected features. The hidden width is solved for
automatically rather than guessed, and a test asserts the match is within 5%:

```python
hidden = match_hidden_width(target_params=sum(p.numel() for p in gcn_branch.parameters()))
```

Without that control, removing the graph branch cannot distinguish a benefit of graph
structure from a benefit of added parameters.

---

## Hyperparameters

All in `configs/default.yaml`; override with `--set key.path=value`.

| Group | Parameter | Value |
|---|---|---|
| input | image size | 320 × 320 (→ 10×10 map → 100 nodes) |
| preprocessing | CLAHE clip limit / tile grid | 2.0 / 8×8, applied at **native** resolution |
| backbone | EfficientNetB0, trainable blocks | last 6 of 9 |
| graph | grid / connectivity / edges | 10×10 / 8 / 684 directed |
| graph branch | layers / hidden / dropout | 3 / 256 / 0.3 |
| readout | pooling | concatenated mean ‖ max → 512-d |
| optimisation | optimizer | AdamW, differential LRs |
| | backbone LR / head LR | 1e-4 / 1e-3 |
| | weight decay | 1e-4 |
| | schedule | 3-epoch linear warmup + single cosine decay, 45 epochs |
| | loss | class-weighted CE, label smoothing 0.05 |
| | auxiliary loss weight $\lambda_\mathrm{aux}$ | 0.4 |
| | batch size / grad clip | 24 / 5.0 |
| | checkpoint selection | best validation **macro F1** |
| | early stopping | patience 10 |
| inference | fusion weight $w$ | 0.6 graph / 0.4 auxiliary |
| | test-time augmentation | horizontal-flip averaging |

Two choices are worth stating explicitly. The checkpoint is selected on macro F1 rather
than accuracy because the validation partition is naturally imbalanced and accuracy can
favour a model that neglects the minority Osteopenia class. And CLAHE is applied
*before* resizing: enhancing first preserves the fine trabecular striations that carry
the diagnostic signal, whereas resizing first would let CLAHE amplify resampling
artefacts instead.

---

## Requirements

Python ≥ 3.9, PyTorch ≥ 2.0, torchvision, scikit-learn ≥ 1.2, OpenCV, Albumentations,
SciPy, matplotlib, PyYAML, Pillow.

Optional: `lime` + `scikit-image` for the LIME panel (`.[xai]`); `torch-geometric` only
to run the equivalence test (`.[pyg]`) — the model itself does not import it;
`requests` + `kagglehub` for the Kaggle path (`.[kaggle]`).

---

## Known limitations

Carried over from the manuscript, and worth stating in the repository too:

- **Single-centre.** One publicly distributed corpus, no multi-site validation, so
  generalisation across scanners and acquisition protocols is unverified.
- **No patient identifiers**, so patient-level leakage cannot be excluded (see above).
- **Labels taken as given.** Severity grades were not confirmed against DXA-derived
  *T*-scores, so label noise at exactly the boundaries the model confuses most cannot
  be ruled out.
- **Osteopenia is evaluated on 57 images.** Its recall interval is correspondingly
  wide, and conclusions about that class are weak.
- **Explanations are not clinically validated.** Deletion/insertion AUCs are reported
  against a random-order control, and failure cases are explained alongside correct
  ones, but no radiologist has reviewed the maps.
- **The severity grades are modelled as nominal.** A cross-entropy objective receives no
  signal that confusing Normal with Osteoporosis is worse than either–Osteopenia; the
  extreme-boundary errors are the direct consequence.
- **Some choices remain unablated**, notably the 320×320 input resolution and the
  random-erasing augmentation.

---

## Citation

```bibtex
@article{sintheia2026efficientnetgcn,
  title   = {EfficientNet-GCN: A Hybrid Convolutional--Graph Framework for Explainable
             Multi-Class Osteoporosis Severity Classification from Knee Radiographs},
  author  = {Sintheia, Tasnim Sultana and Muin, Musfique Rahman and
             Gupta, Pranta Das and Mobin, Md. Iftekharul},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The knee radiograph dataset is subject to its own terms
and is not covered by this licence.

> **Not a diagnostic device.** The framework is positioned as a triage aid — flagging
> radiographs that warrant DXA referral — not as a replacement for dual-energy X-ray
> absorptiometry. No deployment claim is supported by a single-centre study.
