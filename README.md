# MAESTRO

<div align="center">
  <img src="assets/image.png" alt="MAESTRO Architecture" width="300px"/>
</div>

**MA**sked **E**ncoding **S**et **TR**ansformer with self-distillati**O**n — a self-supervised model that learns a single fixed-length representation of an entire cytometry (CyTOF/flow) sample directly from its unordered set of single cells, without requiring cell-level labels.

Each patient sample is a *set* of tens of thousands of cells, each described by a vector of marker intensities. MAESTRO encodes that whole set into one latent embedding per sample. Those embeddings can then be used for downstream tasks such as patient-level diagnosis classification, regression on clinical variables, and cell-population interpretation.

---

## Table of contents

1. [System requirements](#1-system-requirements)
2. [Installation guide](#2-installation-guide)
3. [Demo](#3-demo)
4. [Instructions for use (your own data)](#4-instructions-for-use-your-own-data)
5. [Reproducing the manuscript results](#5-reproducing-the-manuscript-results)
6. [Repository structure](#6-repository-structure)
7. [External references](#7-external-references)
8. [License](#8-license)
9. [Citation & contact](#9-citation--contact)

---

## 1. System requirements

### Software dependencies

MAESTRO is a Python package. The core dependencies and the versions it has been **tested on** are:

| Dependency | Tested version | Notes |
|---|---|---|
| Python | 3.10 and 3.12 | |
| PyTorch | 2.5.1 | install method depends on your OS/GPU (see below) |
| Lightning / pytorch-lightning | 2.4.0 | training loop & checkpointing |
| DeepSpeed | 0.16.3 | distributed / multi-GPU training |
| geomloss | latest on PyPI | Sinkhorn / energy reconstruction loss |
| entmax | latest on PyPI | sparse attention |
| pykeops | latest on PyPI | kernel operations for geomloss |
| umap-learn | 0.5.7 | embedding visualization |
| scikit-learn | 1.6.1 | downstream classifiers/regressors |
| numpy | 1.26.4 | |
| pandas | 2.2.3 | |
| h5py | latest on PyPI | reading the `.h5` sample files |

A complete pinned environment that mirrors the development machine is provided in [`environment.yml`](environment.yml) (Linux/CUDA 12.4). A minimal, OS-agnostic list is in [`requirements.txt`](requirements.txt).

### Operating systems

- **Tested on:** Ubuntu Linux (CUDA 12.4) for full training, and macOS (Apple Silicon, MPS backend) for the post-training analysis notebook.
- Should also run on other Linux distributions and on CPU-only machines, though training without a GPU is impractically slow (see hardware note below).

### Hardware requirements

- **Training:** A CUDA-capable NVIDIA GPU is strongly recommended (and required to reproduce the manuscript results at scale). The code uses DeepSpeed and supports multiple GPUs. No other non-standard hardware is required.
- **Post-training analysis:** The provided pre-trained demo checkpoint and the analysis notebook run on a normal desktop/laptop CPU (or Apple Silicon MPS) in a few minutes — no GPU needed.

---

## 2. Installation guide

> **Typical install time:** about **10–20 minutes** on a normal desktop computer with a reasonable internet connection (most of the time is spent downloading PyTorch and its CUDA libraries).

### Step 1 — Create a conda environment

```bash
conda create -n maestro python=3.10 -y
conda activate maestro
```

### Step 2 — Install PyTorch

PyTorch must be installed **before** the other dependencies, because the correct build depends on your operating system and GPU. Pick the line that matches your machine (or get the exact command for your setup from [pytorch.org/get-started](https://pytorch.org/get-started/locally/)):

```bash
# macOS (Apple Silicon or Intel — CPU only)
pip install torch torchvision torchaudio

# Linux with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Linux with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Linux, CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify the installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from models.MAESTRO import MAESTRO; print('MAESTRO imported successfully')"
```

On macOS (Apple Silicon) you can additionally confirm the MPS backend is available:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If both commands print without error, the installation succeeded.

### Alternative: exact Linux/GPU reproduction

If you are on **Linux with CUDA 12.4** and want to reproduce the exact development environment (pinned versions and builds), you can skip the steps above and create the environment directly from the export:

```bash
conda env create -f environment.yml
conda activate maestro_rapids_env
```

> This file is platform-specific (Linux/CUDA 12.4) and will not resolve on macOS or CPU-only machines — use the cross-platform `requirements.txt` path above for those.

---

## 3. Demo

A small, fully self-contained demo dataset ships with this repository so you can run MAESTRO end-to-end without downloading anything. **All data here is simulated** (no real patient data) and is provided purely to demonstrate the software.

### The demo dataset

`data/` contains 100 synthetic patient samples split across two CyTOF "panels":

| Folder | Samples | Markers per cell | Cells per sample |
|---|---|---|---|
| `data/csv/dataA`, `data/h5/dataA` | 50 (`patient_001`–`patient_050`) | 30 | ~15,000–30,000 |
| `data/csv/dataB`, `data/h5/dataB` | 50 (`patient_101`–`patient_150`) | 30 | ~15,000–30,000 |

The two panels share their first 20 markers and differ in the rest — this demonstrates MAESTRO's ability to train across panels with **different marker sets** by automatically using only the shared markers. `data/csv/metadata.csv` provides per-patient `Diagnosis`, `Age`, `Sex`, and `Treatment` labels used by the downstream analyses.

- `.csv` files are the human-readable source data (one row per cell; last column is the gated cell type).
- `.h5` files are the model-ready version and are **already provided**. If you want to regenerate them (or convert your own CSVs), run the notebook [`data/convert_csv_to_h5.ipynb`](data/convert_csv_to_h5.ipynb).

### Demo Part A — Train a model on the demo data

From the repository root:

```bash
bash train_maestro.sh
```

This launches training on `data/h5/dataA`. To make the demo finish quickly, open [`train_maestro.sh`](train_maestro.sh) and lower the epoch count, e.g. change `--epochs 1000` to `--epochs 20`.

**Expected output:**
- Console logs reporting the inferred shared markers, the number of training samples, and a per-epoch line with the loss (reconstruction + self-distillation), e.g.
  `🎶 Epoch 5 [energy] | Duration: 0.4 min | Loss: ... | Recon: ... | Distillation: ... 🎶`
- A new `output/<project_name>/` directory containing:
  - `config.pth` — the model configuration,
  - periodic and best checkpoints (`*.ckpt`),
  - `reconstruction_viz/epoch_XXXX.pdf` — UMAP visualizations comparing original vs. reconstructed cells, written every 10 epochs.

**Expected run time:** roughly **0.3–1 minute per epoch on a single modern GPU** for this demo size; a 20-epoch demo run therefore takes only a few minutes. A full 1000-epoch run takes several hours and requires a GPU. *(These figures are hardware-dependent estimates — verify on your own machine.)*

> DeepSpeed checkpoints are sharded. To consolidate a `last.ckpt/` directory into a single weights file, see [`output/training/make_model.sh`](output/training/make_model.sh):
> ```bash
> python -u output/training/ToyModel/last.ckpt/zero_to_fp32.py \
>     output/training/ToyModel/last.ckpt/ output/training/ToyModel/model.ckpt
> ```

### Demo Part B — Downstream analysis with a pre-trained model

You don't need a GPU (or even a freshly trained model) for this part. A small pre-trained checkpoint, `ToyModel`, is included under `output/training/ToyModel/`.

Launch Jupyter and open the analysis notebook:

```bash
jupyter notebook notebook/post_analysis.ipynb
```

Run the cells top to bottom. The notebook (which already points at `output/training/ToyModel/model.ckpt`) demonstrates:

1. Loading the trained model and the demo data,
2. **Reconstruction** of masked cells (UMAP before/after),
3. Extracting one **latent embedding per patient**,
4. UMAP of the patient-level latent space colored by metadata,
5. **Diagnosis classification** (logistic regression, 5-fold CV),
6. **Sex prediction** and **age regression**,
7. **Cell-type-proportion prediction**,
8. **Batch-effect / stability analysis**,
9. **Pooling-attention interpretation** (which cell types the model attends to).

**Expected output:** inline UMAP scatter plots, confusion matrices, and printed accuracy / F1 / R² metrics for each task.

**Expected run time:** a few minutes total on a normal CPU or Apple Silicon laptop.

---

## 4. Instructions for use (your own data)

### 1. Format your data

Arrange each sample as a CSV with one row per cell. Every column is a marker intensity, **except the final column** which must be named `CellType` (the gated cell-type label; if you have no labels you can fill this with a placeholder). Place all samples for a panel in their own folder, e.g. `data/csv/myStudy/`.

### 2. Convert CSV → H5

Edit the `DATASETS` list at the top of [`data/convert_csv_to_h5.ipynb`](data/convert_csv_to_h5.ipynb) to include your folder name, then run the notebook. It writes one `.h5` file per sample (containing `data`, `cell_types`, and `feature_names`) into `data/h5/myStudy/`. Samples with fewer than `MIN_CELLS` (default 5000) cells are skipped.

### 3. Train

Point `--data_dirs` at your H5 folder(s):

```bash
python -u train_maestro.py \
    --project 'MyStudy' \
    --devices '0' \
    --data_dirs ./data/h5/myStudy \
    --epochs 1000 \
    --mode 'Train'
```

- Pass several folders to `--data_dirs` (e.g. `./data/h5/dataA ./data/h5/dataB`) to train across multiple panels at once; MAESTRO automatically uses only the markers shared by all of them.
- Use `--marker_dirs` to include a panel's markers in the shared-marker intersection *without* training on its samples.
- For multi-GPU training, set `--devices '0,1,2,3'`.
- Run `python train_maestro.py --help` to see all hyperparameters (hidden/latent dimensions, number of inducing points, temperatures, learning rate, etc.).

### 4. Analyze

Copy `notebook/post_analysis.ipynb`, update the `CHECKPOINT_PATH` / `CONFIG_PATH` near the top to your trained model (under `output/MyStudy/`), and update the data directory paths.

---

## 5. Reproducing the manuscript results

The downstream analyses reported in the manuscript (diagnosis classification, age/sex prediction, cell-type-proportion prediction, batch-effect analysis, and pooling-attention interpretation) are all reproduced cell-by-cell in [`notebook/post_analysis.ipynb`](notebook/post_analysis.ipynb), run against a trained checkpoint. A detailed description of the model and training procedure (pseudocode-level) is provided in the **Methods** section of the manuscript.

---

## 6. Repository structure

```
MAESTRO/
├── train_maestro.py        # Main training entry point (argument-driven)
├── train_maestro.sh        # Example training command for the demo data
├── requirements.txt        # Minimal, OS-agnostic dependency list
├── environment.yml         # Full pinned conda environment (Linux/CUDA 12.4)
│
├── configs/
│   └── config.py           # DeepSpeed config + training callbacks
│
├── data/
│   ├── cytof_dataset.py    # CyTOFDataset: loads .h5 samples, computes shared markers
│   ├── convert_csv_to_h5.ipynb  # CSV → H5 conversion + validation
│   ├── csv/                # Demo data (human-readable source CSVs) + metadata.csv
│   └── h5/                 # Demo data (model-ready, pre-converted)
│
├── models/
│   └── MAESTRO.py          # Model: Set Transformer encoder/decoder, masking,
│                           #   student–teacher self-distillation, Lightning module
│
├── notebook/
│   └── post_analysis.ipynb # End-to-end downstream analysis / reproduction notebook
│
├── output/
│   └── training/ToyModel/  # Small pre-trained checkpoint for the analysis demo
│
└── assets/
    └── image.png           # Architecture figure
```

---

## 7. External references

- DeepSets implementation inspired by: https://github.com/manzilzaheer/DeepSets
- Set Transformer implementation inspired by: https://github.com/juho-lee/set_transformer

---

## 8. License

This project is released under the [MIT License](LICENSE), an [Open Source Initiative](https://opensource.org/licenses/MIT)–approved license. See the [`LICENSE`](LICENSE) file for the full text.

---

## 9. Citation & contact

**Author:** Matthew E. Lee — `matthew.lee1@pennmedicine.upenn.edu`
**Advisors:** E. John Wherry & Dokyoon Kim

If you use MAESTRO in your work, please cite the accompanying manuscript (citation to be added upon publication).
