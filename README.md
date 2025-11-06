# Froth SAM Family

Unified fine-tuning and evaluation scripts for **SAM**, **HQ-SAM**, and **MedSAM** on a proprietary froth segmentation task.

Authors: **Sina Lotfi**, **Reza Dadbin**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Configuration](#configuration)
6. [Data Preparation](#data-preparation)
7. [Running the Pipeline](#running-the-pipeline)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Soft Mask Prediction](#soft-mask-prediction)
   - [AMG-Style Visualization](#amg-style-visualization)
8. [Outputs](#outputs)
9. [Design Notes & Limitations](#design-notes--limitations)
10. [References](#references)

---

## Project Overview

This repository delivers a **script-based pipeline** for fine-tuning Segment Anything–style models on a froth segmentation dataset. The pipeline supports three ViT-B backbones:

- **SAM** (Segment Anything Model)
- **HQ-SAM** (High-Quality SAM)
- **MedSAM** (Medical SAM, adapted for this domain)

All training is performed in a **decoder-only** fashion using a **full-image box prompt**. The encoders remain frozen to maintain training stability while adapting the mask decoder to froth-specific imagery.

> ⚠️ **Dataset availability**
> The froth dataset is **private** and **not distributed** with this repository. You must supply your own dataset following the [expected structure](#data-preparation).

## Key Features

- Unified CLI scripts for training, evaluation, prediction, and visualization across three SAM-family models.
- Modular Python package (`sam_froth`) containing dataset loaders, model wrappers, loss functions, and metrics.
- Lightweight `config.py` to centralize paths, hyperparameters, and checkpoint locations.
- Decoder-only fine-tuning strategy for efficient adaptation without retraining large encoders.
- Utilities for exporting soft masks and generating AMG-style qualitative overlays.

## Repository Structure

```text
froth-sam-family/
├─ config.py                 # Central configuration values
├─ requirements.txt          # Python dependencies
├─ README.md                 # Project documentation
│
├─ sam_froth/                # Package with shared components
│   ├─ data/
│   │   └─ froth_dataset.py  # TIFF + LabelMe JSON dataset loader
│   ├─ models/
│   │   ├─ sam_base.py       # Base SAM helpers
│   │   ├─ hqsam.py          # HQ-SAM helpers
│   │   └─ medsam.py         # MedSAM helpers
│   └─ utils/
│       ├─ losses.py         # BCEDiceLoss implementation
│       └─ metrics.py        # IoU, Dice, and related metrics
│
├─ scripts/                  # Entry points for the pipeline
│   ├─ train.py              # Fine-tune decoder for chosen model
│   ├─ eval.py               # Compute quantitative metrics
│   ├─ predict.py            # Export continuous mask predictions
│   └─ amg_demo.py           # AMG-style qualitative visualization
│
├─ data/                     # (git-ignored) place your dataset here
│   ├─ train/
│   ├─ test/
│   └─ eval/
│
├─ weights/                  # (git-ignored) pretrained & finetuned weights
└─ outputs/                  # (git-ignored) experiment artifacts
```

## Environment Setup

### Python & PyTorch

- Python **3.10** or newer is recommended.
- Install a PyTorch build compatible with your CUDA/cuDNN stack (CPU-only builds are also supported).

### Python Dependencies

1. Create and activate a virtual environment (optional but recommended).
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   The list includes (non-exhaustive): `torch`, `torchvision`, `numpy`, `tifffile`, `opencv-python`, `matplotlib`, `tqdm`, `segment-anything`, and `segment-anything-hq`.

### Optional GPU Setup

Ensure your environment exposes the correct CUDA devices if you intend to train or run inference on a GPU. Scripts respect standard PyTorch CUDA environment variables.

## Configuration

Edit **`config.py`** to customize:

- Dataset root directories
- Output directories for checkpoints, logs, and predictions
- Training hyperparameters (batch size, epochs, learning rate)
- Model-specific checkpoint paths (`sam_vit_b_01ec64.pth`, `sam_hq_vit_b.pth`, `medsam_vit_b.pth`)

All CLI scripts import from `config.py`, so updates automatically propagate across the pipeline.

## Data Preparation

The code expects paired TIFF images and LabelMe-style JSON annotations. Organize your dataset as follows:

```text
data/
├─ train/
│   ├─ image_001.tif
│   ├─ image_001.json
│   ├─ image_002.tif
│   └─ image_002.json
├─ test/
│   └─ ...
└─ eval/                      # optional hold-out split
    └─ ...
```

Each JSON file should contain polygons delineating froth regions. During training the scripts automatically convert these annotations to binary masks aligned with the TIFF imagery.

## Running the Pipeline

All commands can be executed from the repository root. Replace `MODEL` with one of `sam`, `hqsam`, or `medsam` as needed.

### Before You Run Any Script

1. **Activate your environment** – ensure the virtual environment (or Conda env) that contains the repository dependencies is active.
2. **Verify paths in `config.py`** – double-check dataset, checkpoint, and output directories so the scripts resolve the correct locations.
3. **Stage your data** – confirm the `data/` directory follows the expected `train/`, `test/`, and optional `eval/` structure with paired TIFF and JSON files.
4. **Download model weights** – place the SAM-family checkpoints referenced in `config.py` under the configured `weights/` directory.
5. **Warm up CUDA (optional)** – if you are using a GPU, run a quick PyTorch tensor allocation to pre-initialize CUDA context and avoid a cold-start delay on the first script execution.

Each CLI run produces log output in the terminal; detailed artifacts (checkpoints, metrics, predictions) are written under the `outputs/` directory tree.

### Training

Fine-tune the decoder of a specific backbone:

```bash
python -m scripts.train --model {sam|hqsam|medsam} --epochs 50
```

**Arguments**

- `--model` – choose `sam`, `hqsam`, or `medsam`.
- `--epochs` – set the total number of fine-tuning epochs.

**Typical workflow**

1. Monitor the console to confirm the script locates the correct backbone weights and dataset split.
2. Watch the training progress bar for loss, Dice, and IoU metrics; these values are also mirrored to `train_log.json` within the model-specific output directory.
3. After training completes, inspect `outputs/<model>_finetune_out/` for the checkpoint with the lowest validation loss (saved as `best_model.pth`).

**Tips**

- Resume training by reusing the same output directory; the script loads the latest checkpoint automatically.
- Adjust batch size or learning rate directly in `config.py` if you encounter GPU memory constraints.

### Evaluation

Compute mean IoU, Dice score, and related metrics on a dataset split:

```bash
python -m scripts.eval --model {sam|hqsam|medsam} --thr 0.5
```

**Arguments**

- `--model` – choose `sam`, `hqsam`, or `medsam` for the decoder being evaluated.
- `--thr` – probability threshold (default `0.5`) used to binarize soft masks before metrics.

**What happens during evaluation**

1. The script loads the latest checkpoint from `outputs/<model>_finetune_out/` unless another path is provided in `config.py`.
2. Predictions are generated for the designated evaluation split and thresholded at the provided probability.
3. Per-image metrics and aggregate statistics (Dice, IoU, precision, recall) are printed and saved to `metrics.json` for later review.

**Verification steps**

- Compare the printed metrics with previous runs to track improvement trends.
- Optionally visualize the confusion mask overlays saved to `outputs/<model>_finetune_out/qualitative_eval/` (enable via configuration).

### Soft Mask Prediction

Generate per-pixel probability maps (0–1) and save them as PNG files:

```bash
python -m scripts.predict --model {sam|hqsam|medsam}
```

**Arguments**

- `--model` – choose `sam`, `hqsam`, or `medsam`.

**Output layout**

- Soft masks (float tensors) are exported alongside 8-bit PNG visualizations under `outputs/pred_masks/<model>_vit_b_decoder_only/soft/`.
- File names mirror the input image stems, making it easy to line up predictions with source data.

**Usage suggestions**

- Convert the saved tensors into TIFF or NIfTI format if you require integration with medical-imaging pipelines.
- Batch multiple prediction runs by pointing `config.py` to different dataset subsets (e.g., `test` vs. `unlabeled`).

### AMG-Style Visualization

Produce qualitative overlays using SAM-style automatic mask generation:

```bash
python -m scripts.amg_demo --model {sam|hqsam|medsam} --split data/split_name --idx 0
```

**Arguments**

- `--model` – choose `sam`, `hqsam`, or `medsam` to match the weights in use.
- `--split` – path or alias to the dataset split AMG should draw from (e.g., `test`, `eval`, or a folder path).
- `--idx` – integer index of the sample within the chosen split.

**Interpretation guide**

- The script renders the original image, predicted masks, and contour overlays in a Matplotlib window (or saves them if headless mode is configured).
- HQ-SAM leverages a grid-based AMG variant that tends to produce denser proposals; expect more candidate masks than with vanilla SAM.
- MedSAM AMG is experimental—treat the results as a qualitative sanity check rather than a definitive segmentation.

**Troubleshooting**

- If the window does not appear, ensure you are running with an available display backend (`matplotlib.use("Agg")` saves to disk instead).
- Large TIFFs might require additional RAM; consider downsampling via `config.py` visualization options if you encounter memory pressure.

## Outputs

All artifacts are stored under `outputs/` (path configurable). Typical contents include:

- `*_finetune_out/` – training logs, checkpoints, and configuration snapshots.
- `pred_masks/` – soft mask PNGs grouped by model.

Keep the directory structure consistent to simplify downstream analysis and reproducibility.

## Design Notes & Limitations

- **Dataset privacy** – No sample data is bundled. Users must prepare their own froth segmentation dataset.
- **Decoder-only fine-tuning** – Encoders remain frozen; adapting them would require script modifications.
- **Binary segmentation** – The current setup assumes a single foreground class. Extending to multi-class would require updates to the dataset processing, loss functions, and decoder heads.
- **MedSAM AMG** – Visualization support exists but is less stable than SAM/HQ-SAM AMG.

## References

- [Segment Anything Model (SAM) — Meta AI](https://github.com/facebookresearch/segment-anything)
- [HQ-SAM](https://github.com/ChaoningZhang/HQ-SAM)
- MedSAM original paper and ViT-B checkpoint by its authors

All froth-specific engineering, integration, and experimentation: **Sina Lotfi** and **Reza Dadbin**.