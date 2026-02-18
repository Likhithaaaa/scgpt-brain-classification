# Hierarchical Brain Cell Type Classification with scGPT

A three-level hierarchical classification pipeline for single-cell RNA-seq data, built on top of the pretrained [scGPT](https://github.com/bowang-lab/scGPT) transformer model. Classifies cells progressively from coarse (brain vs non-brain) to fine-grained (297 subtypes collapsed to 22 biological groups).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory Layout](#directory-layout)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
   - [Level 1- Brain vs Non-Brain](#level-1--brain-vs-non-brain)
   - [Level 2- Cell Type Classification](#level-2--cell-type-classification)
   - [Level 3- Subtype Classification](#level-3--subtype-classification)
6. [Inference](#inference)
   - [Level 1 Inference](#level-1-inference)
   - [Level 2 Inference](#level-2-inference)
   - [Level 3 Inference](#level-3-inference)
7. [Output Files Reference](#output-files-reference)
8. [Confidence Rejection System](#confidence-rejection-system)
9. [Geneformer Baseline](#geneformer-baseline)

---

## Overview

The pipeline classifies single-cell RNA-seq data in three consecutive levels:

```
Input: raw .h5ad (scRNA-seq counts)
         |
    [ Level 1 ] Brain vs Non-Brain  (binary, scGPT_human weights, frozen encoder)
         |
    [ Level 2 ] Cell Type           (17 classes, scGPT_brain weights, fine-tuned)
         |
    [ Level 3 ] Subtype             (22 biological groups, scGPT_brain weights, fine-tuned)
```

Each level applies a **confidence-based rejection** mechanism: predictions below a threshold are flagged as ambiguous or escalated to a coarser label rather than returned as wrong guesses.

Training data is the ABC Atlas non-neuron brain dataset (~888k cells, ~59k genes). The non-brain reference is PBMC 10k.

---

## Prerequisites

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB | 40 GB (A100) |
| RAM | 64 GB | 128 GB |
| Disk | 50 GB | 100 GB |

Level 3 training in particular is memory-intensive due to the large number of subtypes. Batch size is set to 3 for Level 3 to compensate.

### Python Environment

Three environment files are provided in `scGPT/`. Use whichever suits your setup:

| File | When to use |
|------|-------------|
| `environment.yml` | **Recommended starting point.** Version-pinned without build strings- works across Linux machines and is easier to resolve. |
| `environment_exact.yml` | Exact build strings included. Use this for bit-for-bit reproducibility on the same platform (Linux x86-64, CUDA 11.7). |
| `requirements.txt` | pip-only install, no conda. Use if you are managing your own environment or are on a system without conda. |

**Key versions** (the environment was developed and tested on Linux with these):

| Package | Version |
|---------|---------|
| Python | 3.10.18 |
| CUDA toolkit | 11.7 |
| PyTorch | 1.13.0 |
| torchtext | 0.14.0 |
| scgpt | 0.2.4 |
| flash-attn | 1.0.4 |
| scanpy | 1.10.3 |
| anndata | 0.8.0 |
| scvi-tools | 0.16.4 |
| numpy | 1.26.4 |
| pandas | 2.3.1 |

**Option A- conda (recommended):**

```bash
# Flexible install (resolves dependencies for your platform)
conda env create -f scGPT/environment.yml

# Or exact reproduction of the original environment (same platform only)
conda env create -f scGPT/environment_exact.yml

conda activate scgpt
```

> The `prefix` field at the bottom of both `.yml` files points to `/home/ubuntu/miniforge/envs/scgpt` (the original DeNBI machine). Conda ignores this when you use `conda env create`; the environment will be installed in your default conda envs directory.

**Option B- pip only:**

```bash
pip install -r scGPT/requirements.txt
```

> flash-attn requires a CUDA-capable GPU and a compatible compiler at install time. If installation fails, install without it and disable the fast transformer backend inside the training scripts by setting `use_fast_transformer=False` in the model constructor calls.

### Model Weights

Inference requires **two kinds of weights**- both are already present in `scGPT/save/` and do not need to be downloaded or retrained:

**1. Pretrained scGPT foundation weights** (used to build the model architecture and initialise the encoder):

```
scGPT/save/
├── scGPT_human/          # Used by Level 1 (brain vs non-brain)
│   ├── best_model.pt     # 196 MB ✓
│   ├── vocab.json        # 1.3 MB ✓
│   └── args.json         # ✓
└── scGPT_brain/          # Used by Levels 2 and 3
    ├── best_model.pt     # 196 MB ✓
    ├── vocab.json        # 1.3 MB ✓
    └── args.json         # ✓
```

**2. Fine-tuned level weights** (the trained classifiers- produced by the training scripts):

```
scGPT/save/
├── scGPT_level1_brain_vs_nonbrain/   # Trained Level 1 classifier ✓
├── scGPT_level2_celltype/            # Trained Level 2 classifier ✓
└── scGPT_level3_subtype/             # Trained Level 3 classifier ✓
```

Each inference script loads the pretrained weights to construct the model, then overwrites the weights with the fine-tuned level checkpoint. **Both must be present for inference to work.**

> **When to retrain:** The existing trained weights are ready to use as-is. Retraining is only needed if you change the training configuration- for example, using a different number of HVGs (`n_hvg`), a different dataset, or different hyperparameters. Changing `n_hvg` in particular will invalidate the saved `hvg_genes.pkl` and require re-running both training and inference from scratch.

**Note on git:** All `.pt` weight files are excluded from the repository via `.gitignore` (they total ~6 GB). Download them as described below before running anything.

---

### Downloading the weights

**Pretrained scGPT foundation weights**- download from the [scGPT release page](https://github.com/bowang-lab/scGPT) and place at:
```
scGPT/save/scGPT_human/   ← best_model.pt, vocab.json, args.json
scGPT/save/scGPT_brain/   ← best_model.pt, vocab.json, args.json
```

**Trained level classifiers**- hosted on HuggingFace Hub at [`likhithav11/scgpt-brain-classification`](https://huggingface.co/likhithav11/scgpt-brain-classification).

Download with:
```bash
pip install huggingface_hub
```

Then run this Python script (save as `download_weights.py` and run with `python download_weights.py`):
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="likhithav11/scgpt-brain-classification",
    repo_type="model",
    local_dir="scGPT/save",
)
print("Done! Weights saved to scGPT/save/")
```

The repo is public- no login required.

### Input Data

All input files must be in **AnnData `.h5ad` format**. The raw count matrix should be in `.X` (cells × genes), unnormalized. Gene names must be in `adata.var_names`.

The pipeline was developed and tested with the [ABC Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas) non-neuron dataset (`Nonneuron.h5ad`, ~888k cells) and a PBMC 10k reference (`Reference.h5ad`) for non-brain cells in Level 1.

---

## Directory Layout

After setup, your working directory should look like this:

```
thesis code/
├── scGPT/
│   ├── environment.yml                # Conda environment (version-pinned)
│   ├── environment_exact.yml          # Conda environment (exact builds)
│   ├── requirements.txt               # pip requirements
│   ├── PreProcessing.py               # Data preprocessing notebook
│   ├── Train/
│   │   ├── level_1.py                 # Level 1 training
│   │   ├── level_2.py                 # Level 2 training
│   │   └── level_3.py                 # Level 3 training
│   ├── Inference/
│   │   ├── inference_level1.py        # Level 1 inference
│   │   ├── inference_level2.py        # Level 2 inference
│   │   └── inference_level3.py        # Level 3 inference
│   ├── collapse_subtypes_biological.py
│   ├── validate_dataset.py
│   ├── fix_collapsed_dataset.py
│   └── save/                          # All model weights live here
│       ├── scGPT_human/               # Pretrained foundation weights (human)
│       ├── scGPT_brain/               # Pretrained foundation weights (brain)
│       ├── scGPT_level1_brain_vs_nonbrain/  # Trained Level 1 classifier
│       ├── scGPT_level2_celltype/           # Trained Level 2 classifier
│       └── scGPT_level3_subtype/            # Trained Level 3 classifier
├── Geneformer/
│   ├── gf_level_1.py
│   ├── gf_level_2.py
│   └── gf_level_3.py
└── data/
    ├── Nonneuron.h5ad                 # ABC Atlas brain cells (put here)
    └── Reference.h5ad                 # PBMC reference for Level 1 (put here)
```

All paths in the scripts are relative to the `scGPT/` directory. Run all training and inference scripts from inside `scGPT/`- they reference `./save/` which resolves to `scGPT/save/`.

---

## Data Preprocessing

**Script:** `scGPT/PreProcessing.py` (Jupyter notebook)

Run this once before training to prepare the ABC Atlas data. The notebook:

- Loads `Nonneuron.h5ad` and auto-detects the ABC Atlas column format
- Extracts cell type labels (Level 2) from subtype names by prefix matching (e.g. `Oligo_50_2996` → `Oligodendrocyte`)
- Creates numeric label IDs for all hierarchy levels
- Computes UMAP coordinates
- Outputs to `./data/`:
  - `metadata_L2_L3.csv`- per-cell metadata with cell type and subtype columns
  - `gene_names.csv`- gene vocabulary
  - `hierarchical_mappings.json`- label-to-ID mappings for all levels
  - `adata_processed_L2_L3.h5ad`- processed AnnData ready for training
  - `umap_visualization_L2_L3.png`- UMAP coloured by cell type

After preprocessing, optionally run the subtype collapsing script to map the 296 computational clusters into 22 biologically interpretable groups:

```bash
python scGPT/collapse_subtypes_biological.py
```

And validate data quality before training:

```bash
python scGPT/validate_dataset.py
```

---

## Training

All training scripts use **WandB** for experiment tracking. Set your WandB API key before running:

```bash
wandb login
```

Or disable WandB by setting `use_wandb = False` inside the `Config` class.

Each training script saves its best checkpoint and supporting artifacts to `./save/scGPT_levelX_*/`.

---

### Level 1- Brain vs Non-Brain

**Script:** `scGPT/Train/level_1.py`

Binary classification distinguishing brain cells from non-brain (PBMC) cells. The scGPT encoder is **frozen**- only the 3-layer classifier head is trained.

**Key configuration** (edit at the top of `level_1.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_model` | `./save/scGPT_human` | Pretrained weights to load |
| `epochs` | 5 | Training epochs |
| `batch_size` | 8 | Micro-batch size |
| `gradient_accumulation_steps` | 16 | Effective batch = 8 × 16 = 128 |
| `lr` | 5e-5 | Learning rate |
| `n_hvg` | 1200 | Highly variable genes selected |
| `n_bins` | 51 | Expression discretisation bins |
| `freeze` | True | Encoder weights are frozen |
| `amp` | True | Mixed precision (AMP) enabled |
| `save_dir` | `./save/scGPT_level1_brain_vs_nonbrain/` | Checkpoint output |

**Run:**

```bash
cd "thesis code/scGPT"
python Train/level_1.py
```

**Artifacts saved to `./save/scGPT_level1_brain_vs_nonbrain/`:**

- `best_model.pt`- best checkpoint by validation accuracy
- `hvg_genes.pkl`- list of HVG gene names (required for inference)
- `id_to_label.json`- integer → label mapping (`{0: "non_brain", 1: "brain"}`)
- Confusion matrices, training curves, UMAP embeddings (PNG files)

---

### Level 2- Cell Type Classification

**Script:** `scGPT/Train/level_2.py`

Classifies brain cells into cell types (e.g. Astrocyte, Oligodendrocyte, Microglia, Endothelial). The scGPT encoder is **fine-tuned** (not frozen), starting from the brain-pretrained weights.

**Key configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_model` | `save/scGPT_brain` | Brain pretrained weights |
| `epochs` | 6 | Training epochs |
| `batch_size` | 8 | Micro-batch size |
| `n_hvg` | 1500 | Highly variable genes |
| `n_bins` | 51 | Expression bins |
| `freeze` | False | Full fine-tuning |
| `max_seq_len` | 1501 | Tokenised sequence length (n_hvg + 1) |

**Run:**

```bash
cd "thesis code/scGPT"
python Train/level_2.py
```

**Artifacts saved to `./save/scGPT_level2_celltype/`:**

- `best_model.pt`
- `hvg_genes.pkl`
- `id_to_label.json`
- `taxonomy.json`- cell type family groupings for hierarchical rejection

---

### Level 3- Subtype Classification

**Script:** `scGPT/Train/level_3.py`

Fine-grained classification into 22 biological subtype groups. Builds on Level 2 and uses the brain-pretrained encoder. Class imbalance is handled with a `WeightedRandomSampler`.

**Key configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_model` | `save/scGPT_brain/best_model.pt` | Starting checkpoint |
| `epochs` | 8 | Training epochs (with early stopping) |
| `batch_size` | 3 | Reduced due to memory requirements |
| `n_hvg` | 1500 | Highly variable genes |
| `n_bins` | 51 | Expression bins |
| `freeze` | False | Full fine-tuning |
| `l3_confidence_threshold` | 0.70 | Internal rejection to Level 2 |
| `l2_confidence_threshold` | 0.65 | Internal rejection to Level 1 |

**Run:**

```bash
cd "thesis code/scGPT"
python Train/level_3.py
```

**Artifacts saved to `./save/scGPT_level3_subtype/`:**

- `best_model.pt`
- `hvg_genes.pkl`
- `id_to_label.json`
- `taxonomy.json`- subtype → parent cell type → family mappings

---

## Inference

Inference scripts load a trained checkpoint and run predictions on any `.h5ad` file containing raw counts. They mirror the exact preprocessing and tokenisation from the corresponding training script.

All three scripts share the same CLI interface:

```
--input       Path to input .h5ad file (required)
--output      Path for output .csv file (required)
--device      cuda / cpu / auto  (default: auto)
--preprocessed  Skip normalization/binning if data is already preprocessed
```

Run inference scripts from the `scGPT/` directory (paths in the scripts point to `./save/`).

---

### Level 1 Inference

```bash
cd "thesis code/scGPT"

# Raw counts (recommended)
python Inference/inference_level1.py --input data/cells.h5ad --output results/results_l1.csv

# Already normalized and binned
python Inference/inference_level1.py --input data/preprocessed.h5ad --output results/results_l1.csv --preprocessed

# Force CPU
python Inference/inference_level1.py --input data/cells.h5ad --output results/results_l1.csv --device cpu
```

**What it does internally:**
1. Loads `hvg_genes.pkl` and subsets input to the 1,200 training HVGs (fills zeros for missing genes)
2. Normalizes to 10k counts per cell, log1p transforms, bins into 51 bins
3. Passes each cell through the scGPT_human encoder + classifier head
4. Applies softmax and confidence threshold τ = 0.45

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `cell_idx` | Row index of the cell in the input file |
| `l1_pred` | Final prediction: `brain`, `non_brain`, or `ambiguous` |
| `l1_raw_pred` | Raw argmax prediction before rejection |
| `l1_confidence` | Max softmax probability (0–1) |
| `l1_brain_prob` | Softmax probability for the brain class |
| `l1_decision` | `accepted` or `ambiguous` |
| `is_brain` | Boolean: True if cell was predicted as brain |

---

### Level 2 Inference

```bash
cd "thesis code/scGPT"

python Inference/inference_level2.py --input data/cells.h5ad --output results/results_l2.csv
```

Best run on cells already identified as brain by Level 1 (filter `is_brain == True` from the Level 1 CSV and re-export as `.h5ad`), though it can be run independently.

**What it does internally:**
1. Subsets to the 1,500 training HVGs
2. Normalizes, log1p, bins (51 bins)
3. Tokenises using `tokenize_and_pad_batch` with `max_len=3001`, `include_zero_gene=False`
4. Runs the scGPT_brain encoder + fine-tuned classifier
5. Applies two-level hierarchical rejection (see [Confidence Rejection System](#confidence-rejection-system))

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `cell_idx` | Row index in input file |
| `l2_pred` | Cell type label, `Family:<name>`, or `rejected` |
| `l2_confidence` | Confidence score used for the decision |
| `l2_decision` | `specific`, `family`, or `rejected` |

---

### Level 3 Inference

```bash
cd "thesis code/scGPT"

python Inference/inference_level3.py --input data/cells.h5ad --output results/results_l3.csv
```

**What it does internally:**
1. Subsets to the 1,500 training HVGs
2. Normalizes, log1p, bins (51 bins)
3. Tokenises with `max_len=1501`
4. Runs scGPT_brain encoder + subtype classifier
5. Applies three-level hierarchical rejection

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `cell_idx` | Row index in input file |
| `l3_pred` | Subtype label, `Parent:<cell_type>`, `Family:<family>`, or `rejected` |
| `l3_confidence` | Confidence score used for the decision |
| `l3_decision` | `specific_subtype`, `parent_type`, `family`, or `rejected` |

---

## Output Files Reference

After running all three inference scripts and merging on `cell_idx`, each cell gets a row like:

```
cell_idx | l1_pred | l1_confidence | l2_pred           | l2_confidence | l3_pred                  | l3_confidence
---------|---------|---------------|-------------------|---------------|--------------------------|---------------
0        | brain   | 0.97          | Oligodendrocyte   | 0.91          | Oligo_late_mature        | 0.88
1        | brain   | 0.83          | Astrocyte         | 0.73          | Family:Astrocyte         | 0.68
2        | brain   | 0.61          | Family:Glial      | 0.67          | rejected                 | 0.32
3        | non_brain | 0.95        |-                 |-             |-                        |-
```

Cells flagged as `ambiguous` at Level 1 or `rejected` at Level 2/3 are not wrong predictions- they simply did not meet the confidence threshold and should be treated as uncertain.

---

## Confidence Rejection System

Rather than always returning the top-1 class, every level applies a rejection cascade. Predictions below threshold are coarsened or discarded:

**Level 1 (binary):**
```
max(P) >= 0.45  →  accepted (brain or non_brain)
max(P) <  0.45  →  ambiguous
```

**Level 2 (cell type):**
```
max(P_specific) >= 0.85  →  specific cell type (e.g. "Oligodendrocyte")
max(P_family)   >= 0.65  →  family label       (e.g. "Family:Glial")
otherwise               →  rejected
```
`P_family` is computed by summing softmax probabilities across all cell types that belong to the same biological family.

**Level 3 (subtype):**
```
max(P_subtype) >= 0.85  →  specific subtype    (e.g. "Oligo_late_mature")
max(P_parent)  >= 0.65  →  parent cell type    (e.g. "Parent:Oligodendrocyte")
max(P_family)  >= 0.45  →  family label        (e.g. "Family:Glial")
otherwise              →  rejected
```
`P_parent` and `P_family` are sums over all subtypes belonging to that parent or family group, as defined in `taxonomy.json`.

Thresholds are set conservatively at Level 1 (τ = 0.45) to preserve sensitivity (avoid discarding real brain cells), and more strictly at Levels 2 and 3 where false specificity is more harmful.

---

## Geneformer Baseline

The `Geneformer/` directory contains equivalent training scripts using the [Geneformer](https://huggingface.co/ctheodoris/Geneformer) model (V1-10M, ~10M parameters) as a baseline comparison:

| Script | Task |
|--------|------|
| `gf_level_1.py` | Brain vs Non-Brain |
| `gf_level_2.py` | Cell type classification |
| `gf_level_3.py` | Subtype classification |

Geneformer uses a different tokenisation strategy (rank-ordered gene tokens based on expression level) compared to scGPT's binned continuous expression encoding. The task structure, evaluation metrics, and output formats are identical to the scGPT scripts, making results directly comparable.

Run the Geneformer scripts the same way as the scGPT training scripts. They require the Geneformer pretrained weights from HuggingFace- download and point the path constant at the top of each script to the local directory.
