# Froth SAM Family  
**Fine-tuning SAM / HQ-SAM / MedSAM for froth segmentation**

Authors: **Sina Lotfi**, **Reza Dadbin**

---

## 1. Overview

This repository contains a **unified, script-based pipeline** for fine-tuning and evaluating three Segment-Anything–style models on a **froth segmentation** task:

- **SAM** (Segment Anything Model, ViT-B)
- **HQ-SAM** (High-Quality SAM, ViT-B)
- **MedSAM** (Medical SAM, ViT-B; adapted here for froth)

All three models are trained in a **decoder-only fine-tuning** setup on a **private froth dataset** (not included in this repo). The goal is **semantic segmentation** of froth regions from industrial images, using a **full-image box prompt** during training.

The repo provides:

- Unified code for:
  - `train.py` — fine-tuning SAM, HQ-SAM, MedSAM
  - `eval.py` — quantitative evaluation (mIoU, Dice)
  - `predict.py` — export soft segmentation maps
  - `amg_demo.py` — AMG-style polygon visualization for each model
- A simple `config.py` used to define paths, hyperparameters and model checkpoints.
- A small internal package: `sam_froth` (dataset, models, losses, metrics).

> ⚠️ **Dataset note**  
> The froth dataset is **private** and **not included**. You must provide your own data in the expected format (see below).

---

## 2. Repository Structure

At the top level:

```text
froth-sam-family/
│
├─ config.py
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ sam_froth/
│   ├─ __init__.py
│   ├─ data/
│   │   ├─ __init__.py
│   │   └─ froth_dataset.py        # TIFF + LabelMe JSON -> tensors
│   ├─ models/
│   │   ├─ __init__.py
│   │   ├─ sam_base.py             # SAM loader + helpers
│   │   ├─ hqsam.py                # HQ-SAM loader + helpers
│   │   └─ medsam.py               # MedSAM loader + helpers
│   └─ utils/
│       ├─ __init__.py
│       ├─ losses.py               # BCEDiceLoss
│       └─ metrics.py              # IoU, Dice, etc.
│
├─ scripts/
│   ├─ train.py                    # train --model {sam,hqsam,medsam}
│   ├─ eval.py                     # eval  --model {sam,hqsam,medsam}
│   ├─ predict.py                  # predict soft masks
│   └─ amg_demo.py                 # AMG-style visualization
│
├─ data/                           # (ignored by git)
│   ├─ train/                      # training split (user-provided)
│   ├─ test/                       # test split (user-provided)
│   └─ eval/                       # optional evaluation-only split
│
├─ weights/                        # (ignored by git)
│   ├─ sam_vit_b_01ec64.pth
│   ├─ sam_hq_vit_b.pth
│   └─ medsam_vit_b.pth
│
└─ outputs/                        # finetuned weights, predictions, etc.
    ├─ sam_finetune_out/
    ├─ hqsam_finetune_out/
    └─ medsam_finetune_out/
3. Environment & Dependencies
3.1. Python & PyTorch
Recommended:

Python ≥ 3.10

PyTorch compatible with your CUDA/cuDNN (or CPU-only installation)

3.2. Python packages
Install dependencies from requirements.txt (you may adjust versions):

bash
Copy code
pip install -r requirements.txt
Typical dependencies include (names only, versions omitted):

torch, torchvision

numpy, tifffile

opencv-python

matplotlib

tqdm

segment-anything # official Meta SAM

segment-anything-hq # HQ-SAM

anything else you normally use in this environment

4. Data Format
The code assumes the following directory layout:

text
Copy code
data/
├─ train/
│   ├─ image_001.tif
│   ├─ image_001.json
│   ├─ image_002.tif
│   ├─ image_002.json
│   └─ ...
├─ test/
│   ├─ image_101.tif
│   ├─ image_101.json
│   └─ ...
└─ eval/                        # optional (same format)
    ├─ ...
4.1. Images
Format: .tif / .tiff

They may be:

grayscale,

RGB,

multi-channel stacks.

The loader (FrothSegmentationDataset) converts them to H×W×3 uint8 RGB:

Converts CHW → HWC if necessary.

Converts grayscale → 3-channel.

Scales >8-bit data into [0, 255] and casts to uint8.

Drops any alpha channel.

4.2. Annotations (LabelMe JSON)
For each image image_001.tif, there should be a sibling image_001.json.

Format: LabelMe JSON.

The code looks for polygons with label:

python
Copy code
label == "froth"
All such polygons are filled into a single binary mask (0/255).

If no polygon has label "froth" but shapes exist, the dataset code may optionally fall back to filling all polygons (implementation detail in FrothSegmentationDataset).

5. Checkpoints (weights/)
Place your pretrained base checkpoints in weights/:

text
Copy code
weights/
├─ sam_vit_b_01ec64.pth      # SAM ViT-B (Meta official)
├─ sam_hq_vit_b.pth          # HQ-SAM ViT-B
└─ medsam_vit_b.pth          # MedSAM ViT-B checkpoint
These filenames correspond to config.py defaults:

python
Copy code
sam_checkpoint   = weights_root / "sam_vit_b_01ec64.pth"
hqsam_checkpoint = weights_root / "sam_hq_vit_b.pth"
medsam_checkpoint = weights_root / "medsam_vit_b.pth"
If your files are named differently, update config.py accordingly.

During training, the repo will save finetuned checkpoints under:

text
Copy code
outputs/
├─ sam_finetune_out/
├─ hqsam_finetune_out/
└─ medsam_finetune_out/
Each subfolder will contain:

*_decoder_best.pth, *_decoder_last.pth — decoder-only weights

*_full_best.pth, *_full_last.pth — full model state dicts

6. Configuration (config.py)
config.py defines paths & hyperparameters via a Config class:

Key fields (simplified):

python
Copy code
from pathlib import Path

class Config:
    # Paths
    project_root = Path(__file__).resolve().parent
    data_root    = project_root / "data"
    train_dir    = data_root / "train"
    test_dir     = data_root / "test"
    eval_dir     = data_root / "eval"
    outputs_root = project_root / "outputs"
    weights_root = project_root / "weights"

    # Dataset
    label_key = "froth"

    # Training
    seed        = 1337
    device      = "auto"     # "cuda" | "cpu" | "auto"
    batch_size  = 1          # keep 1 for SAM/HQ-SAM prompts
    num_workers = 0
    pin_memory  = True

    epochs       = 10
    lr           = 1e-5
    weight_decay = 0.0

    # Model selection (filled in by CLI)
    model_name = "sam"             # "sam" | "hqsam" | "medsam"
    train_mode = "decoder_only"    # used in naming checkpoints

    # Base checkpoints
    sam_checkpoint   = weights_root / "sam_vit_b_01ec64.pth"
    sam_model_type   = "vit_b"
    hqsam_checkpoint = weights_root / "sam_hq_vit_b.pth"
    hqsam_model_type = "vit_b"
    medsam_checkpoint = weights_root / "medsam_vit_b.pth"
    medsam_model_type = "vit_b"

    # Finetune output dir (set in setup())
    finetune_out = outputs_root / "sam_finetune_out"

    @classmethod
    def setup(cls):
        # create directories if needed
        cls.outputs_root.mkdir(parents=True, exist_ok=True)
        cls.weights_root.mkdir(parents=True, exist_ok=True)
        cls.data_root.mkdir(parents=True, exist_ok=True)

        # select output subdir based on model
        if cls.model_name.lower() == "sam":
            subdir = "sam_finetune_out"
        elif cls.model_name.lower() == "hqsam":
            subdir = "hqsam_finetune_out"
        else:
            subdir = f"{cls.model_name}_finetune_out"

        cls.finetune_out = cls.outputs_root / subdir
        cls.finetune_out.mkdir(parents=True, exist_ok=True)
How it’s used:

Each script does:

python
Copy code
C.model_name = args.model.lower()
C.setup()
So you normally do not touch C.model_name yourself; you select the model via --model.

If you want to change paths or hyperparameters (e.g. lr, epochs), edit config.py and rerun the scripts.

7. How to Run: Training
All scripts are invoked with python -m scripts.<name> ... from the repo root.

7.1. SAM (ViT-B) training
bash
Copy code
python -m scripts.train --model sam
Uses:

base checkpoint at weights/sam_vit_b_01ec64.pth

data in data/train and data/test

Trains mask decoder only (encoders frozen).

Saves finetuned checkpoints under:

text
Copy code
outputs/sam_finetune_out/
    sam_vit_b_decoder_only_decoder_best.pth
    sam_vit_b_decoder_only_decoder_last.pth
    sam_vit_b_decoder_only_full_best.pth
    sam_vit_b_decoder_only_full_last.pth
(The exact naming pattern is model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}".)

7.2. HQ-SAM training
bash
Copy code
python -m scripts.train --model hqsam
Uses:

base checkpoint at weights/sam_hq_vit_b.pth

Decoder-only finetuning:

HQ encoder opens image & returns img_embed + interm_embeddings.

Decoder is trained while encoder and prompt encoder are frozen.

Saves under:

text
Copy code
outputs/hqsam_finetune_out/
    hqsam_vit_b_decoder_only_decoder_best.pth
    ...
7.3. MedSAM training
bash
Copy code
python -m scripts.train --model medsam
Uses:

base MedSAM checkpoint at weights/medsam_vit_b.pth

Also decoder-only finetuning:

Full-image box prompt is derived from froth mask bounding box.

Postprocessing follows SAM-style 256→1024→original resize logic.

Saves under:

text
Copy code
outputs/medsam_finetune_out/
    medsam_vit_b_decoder_only_decoder_best.pth
    medsam_vit_b_decoder_only_full_best.pth
    ...
ℹ️ Note: Training for all three backends uses the same FrothSegmentationDataset and collate_froth loader logic internally (same data, same mask convention).

8. How to Run: Evaluation
eval.py computes mean IoU and mean Dice over a split.

Priority for evaluation:

data/eval/ (if exists & non-empty)

else data/test/

else data/train/

Common CLI options
--model {sam,hqsam,medsam}

--thr <float> — probability threshold for binarizing masks (default: 0.5)

8.1. SAM
bash
Copy code
python -m scripts.eval --model sam --thr 0.5
Console example:

text
Copy code
Using model: sam | device: cuda
Evaluating on 10 samples from [test] dir: data/test
Loaded SAM finetuned model: sam_vit_b_decoder_only_full_best.pth

[SAM] Eval @ thr=0.50 → mIoU=0.9346, mDice=0.9662
8.2. HQ-SAM
bash
Copy code
python -m scripts.eval --model hqsam --thr 0.5
Example:

text
Copy code
Using model: hqsam | device: cuda
Evaluating on 10 samples from [test] dir: data/test
Loaded HQ-SAM finetuned model: hqsam_vit_b_decoder_only_full_best.pth

[HQ-SAM] Eval @ thr=0.50 → mIoU=..., mDice=...
8.3. MedSAM
bash
Copy code
python -m scripts.eval --model medsam --thr 0.5
Example:

text
Copy code
Using model: medsam | device: cuda
[MedSAM] Loaded via registry: medsam_vit_b.pth
Loaded MedSAM finetuned model: medsam_vit_b_decoder_only_full_best.pth

[MedSAM] Eval @ thr=0.50 → mIoU=..., mDice=...
9. How to Run: Prediction (Soft Segmentation Maps)
predict.py exports soft masks as grayscale PNGs (0–255) for each image in the selected split.

Priority is the same as eval:

data/eval/

data/test/

data/train/

Each backend writes to:

text
Copy code
outputs/pred_masks/<model_tag>/soft/mask_0000.png
where model_tag encodes backend + variant.

9.1. SAM soft masks
bash
Copy code
python -m scripts.predict --model sam
Outputs under something like:

text
Copy code
outputs/pred_masks/sam_vit_b_decoder_only/soft/
    mask_0000.png
    mask_0001.png
    ...
These are continuous probability maps normalized to 0–255.

9.2. HQ-SAM soft masks
bash
Copy code
python -m scripts.predict --model hqsam
Similar directory, but hqsam_* tag.

9.3. MedSAM soft masks
bash
Copy code
python -m scripts.predict --model medsam
Similar directory, but medsam_* tag.

Note: Prediction scripts do not apply postprocessing beyond SAM-style resizing / padding. Additional morphological postprocessing can be applied externally if needed.

10. How to Run: AMG / Visualization
amg_demo.py is for qualitative visualization of model predictions as polygons overlaid on the original image.

CLI:

--model {sam,hqsam,medsam}

--idx <int> — sample index in chosen split

--split {auto,train,test,eval} — which split to draw from

auto chooses eval > test > train based on availability.

10.1. SAM AMG (official SamAutomaticMaskGenerator)
bash
Copy code
python -m scripts.amg_demo --model sam --split test --idx 0
Uses SamAutomaticMaskGenerator from segment_anything.

Draws one colored contour per detected region over the original image.

Shows:

original image

ground truth froth mask

SAM AMG polygons (with contour count)

10.2. HQ-SAM AMG (custom generator)
bash
Copy code
python -m scripts.amg_demo --model hqsam --split test --idx 0
Uses a custom AMG-like generator:

builds a grid of point prompts,

decodes one mask per point,

filters by predicted IoU, probability, area,

applies simple mask NMS,

draws colored contours.

10.3. MedSAM AMG (experimental)
bash
Copy code
python -m scripts.amg_demo --model medsam --split test --idx 0
Uses an experimental AMG-style generator adapted from the SAM/HQ-SAM versions.

Behavior:

It does produce masks and contours,

But results are less stable and less reliable than SAM/HQ-SAM AMG for this froth task.

We recommend treating MedSAM AMG as a visual sanity check only, and relying on:

quantitative metrics from eval.py,

soft maps from predict.py.

11. Notes & Limitations
Dataset not included.
All results depend on a private froth segmentation dataset (TIFF + LabelMe JSON). You must provide your own data in the same format.

Decoder-only fine-tuning.
For all three backends (SAM, HQ-SAM, MedSAM), only the mask decoder (and relevant heads) are trained; encoders and prompt encoders are frozen. This is intentional to keep training stable and lightweight.

Binary segmentation task.
This repo is set up for binary froth vs. background segmentation. Extending to multi-class would require changes to the dataset, losses, and mask decoding logic.

MedSAM AMG is experimental.
MedSAM’s AMG-style generator is not guaranteed to behave like the official SAM AMG. Use it for qualitative visualization, not as a precise instance segmentation tool.

12. Credits & References
This work builds on the following projects (please cite them if you use this repo):

SAM (Segment Anything Model) — Meta AI

GitHub: facebookresearch/segment-anything

HQ-SAM (High-Quality SAM)

GitHub: ChaoningZhang/HQ-SAM

MedSAM (Medical SAM)

Original paper and repo by its authors (ViT-B checkpoint used here).

All froth-specific engineering, integration, and experiment setup:

Sina Lotfi, Reza Dadbin