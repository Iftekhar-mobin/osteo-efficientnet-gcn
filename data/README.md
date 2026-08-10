# Data

The radiographs are **not redistributed here**. They come from a public Kaggle dataset
with its own terms.

## Getting the corpus

**Multi-Class Knee Osteoporosis X-Ray Dataset** — 1,947 knee radiographs in three
diagnostic categories.

<https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset>

```bash
python scripts/00_download_data.py            # via kagglehub, no credentials needed
```

The script also verifies the class counts, which must be exactly:

| Class | Images |
|---|---|
| Normal | 780 |
| Osteopenia | 374 |
| Osteoporosis | 793 |
| **Total** | **1,947** |

If those counts do not match, the dataset has been revised since this work and every
downstream number should be treated as measuring a different corpus.

## Layout

```
data/
├── raw/OS Collected Data/     the three class folders, as distributed (git-ignored)
├── processed/                 320x320 CLAHE-enhanced PNGs + per-partition indices
│   ├── train/ val/ test/
│   └── {train,val,test}_index.json
├── processed_noclahe/         ablation (iv)
├── processed_unbalanced/      ablation (v)
└── splits/manifest.json       the partition manifest -- committed, small, auditable
```

`data/splits/manifest.json` is the one data artefact that **is** committed. It lists
every source filename and the partition it landed in, so the split can be audited
without redistributing a single image.

## On patient identifiers

The distribution carries **no patient-level identifiers**. The partition is verified
free of image-level overlap (`tests/test_protocol_integrity.py`), but if the corpus
contains bilateral pairs or repeat studies from one individual, patient identity can
still cross partitions and the reported metrics would be optimistic. This cannot be
excluded with the metadata available, and it applies equally to every published study
using this corpus.

Several other properties relevant to interpretation are undocumented by the source: the
acquiring institution, patient demographics, acquisition parameters, annotator
qualifications, and whether the severity labels were established against DXA-derived
*T*-scores or by radiographic reading alone.
