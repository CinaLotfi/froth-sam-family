# Froth-SAM Family  
**Froth Segmentation with SAM, HQ-SAM (and MedSAM-ready)**  

Authors: **Sina Lotfi**, **Reza Dadbin**

---

## 1. Overview

This repository contains a **clean, script-based implementation** of froth segmentation using the **Segment Anything family** of models:

- **SAM (ViT-B)** – baseline finetuned model  
- **HQ-SAM (ViT-B)** – high-quality SAM variant with HQ tokens  
- **MedSAM** – planned extension point (hooks already prepared in the codebase)

The goal is **semantic segmentation of froth** in process images, given:

- Input images as **TIFF (`.tif` / `.tiff`)**
- Annotations as **LabelMe `.json` polygons** with label `"froth"`

The repo is organized so that:

- You can **train**, **evaluate**, **predict**, and **visualize AMG-style masks** with *one command per model*.
- All logic is in **Python packages + scripts**, no notebooks required.

---

## 2. Repository Structure

At the top level the repo looks like this:

```text
froth-sam-family/
├─ config.py                 # Central config object
├─ .gitignore
├─ sam_froth/                # Python package
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ froth_dataset.py    # FrothSegmentationDataset + preprocessing/collate
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ sam_base.py         # SAM + HQ-SAM loaders, postprocess helpers, etc.
│  └─ utils/
│     ├─ __init__.py
│     ├─ losses.py           # BCEDiceLoss, etc.
│     └─ metrics.py          # IoU/Dice helpers (if needed)
│
├─ scripts/                  # Entry-point scripts (run from repo root)
│  ├─ train.py               # Train SAM or HQ-SAM (decoder-only / full)
│  ├─ eval.py                # Evaluate mIoU / Dice for SAM / HQ-SAM
│  ├─ predict.py             # Run inference and export soft segmentation maps
│  └─ amg_demo.py            # AMG-style visualization (contours) per image
│
├─ data/                     # NOT tracked in git (private dataset)
│  ├─ train/                 # train TIFF + JSON
│  ├─ test/                  # test TIFF + JSON
│  └─ eval/                  # optional eval TIFF + JSON
│
├─ weights/                  # NOT tracked in git (model checkpoints)
│  ├─ sam_vit_b_01ec64.pth          # base SAM ViT-B checkpoint
│  ├─ sam_hq_vit_b.pth              # base HQ-SAM ViT-B checkpoint
│  └─ ... (MedSAM weights, etc.)
│
└─ outputs/                  # NOT tracked in git (training outputs)
   ├─ sam_finetune_out/      # SAM training logs + checkpoints
   ├─ hqsam_finetune_out/    # HQ-SAM training logs + checkpoints
   └─ pred_masks/            # Predicted soft masks per model
Note: data/, weights/, outputs/ are ignored by git on purpose.
Only code + config + scripts live in the repo.

3. Data Format & Folder Layout
3.1. Expected Directory Layout
Under data/ we expect three splits (you can use only some of them):

text
Copy code
data/
├─ train/
├─ test/
└─ eval/      # optional
Each of these folders should contain pairs:

Image file: *.tif or *.tiff

Annotation: same basename with .json (LabelMe format)

Example (data/train):

text
Copy code
data/train/
├─ img_0001.tiff
├─ img_0001.json
├─ img_0002.tiff
├─ img_0002.json
└─ ...
3.2. Annotation Format (LabelMe JSON)
The dataset uses LabelMe polygons. For each image:

shapes is a list of annotated regions.

Each shape has:

"label": "froth" (or other label; see below)

"points": [[x1, y1], [x2, y2], ..., [xn, yn]]

Example snippet:

json
Copy code
{
  "imagePath": "img_0001.tiff",
  "shapes": [
    {
      "label": "froth",
      "points": [[123.4, 56.7], [130.2, 60.1], [135.0, 59.8]],
      "shape_type": "polygon"
    }
  ]
}
Inside FrothSegmentationDataset:

By default it looks for polygons whose label == "froth".

All "froth" polygons are filled to create a single binary mask:

0 = background

255 = froth

If no "froth" label is found but other polygons exist, the code falls back to filling all polygons.

4. Configuration (config.py)
All paths & high-level hyperparameters are centralized in config.py.

Key things (conceptually):

python
Copy code
from pathlib import Path

class Config:
    # --- paths ---
    project_root = Path(__file__).resolve().parent
    data_root    = project_root / "data"
    train_dir    = data_root / "train"
    test_dir     = data_root / "test"
    eval_dir     = data_root / "eval"

    weights_root = project_root / "weights"
    outputs_root = project_root / "outputs"

    # SAM checkpoints
    sam_checkpoint   = weights_root / "sam_vit_b_01ec64.pth"
    sam_model_type   = "vit_b"

    # HQ-SAM checkpoints
    hqsam_checkpoint = weights_root / "sam_hq_vit_b.pth"
    hqsam_model_type = "vit_b"

    # training / dataloader defaults
    batch_size   = 1              # kept at 1 for SAM box prompts; we simulate larger batches via accumulation
    num_workers  = 0
    pin_memory   = True
    seed         = 1337
    device       = "auto"         # "auto" | "cuda" | "cpu"

    # label name in JSON
    label_key    = "froth"

    # train mode: "decoder_only", "decoder+prompt", "encoder_only", "prompt_only", "full"
    train_mode   = "decoder_only"

    # per-run model name, set from CLI (--model sam|hqsam)
    model_name   = "sam"

    @classmethod
    def setup(cls):
        # Create outputs/ subfolder depending on model_name
        cls.finetune_out = cls.outputs_root / f"{cls.model_name}_finetune_out"
        cls.finetune_out.mkdir(parents=True, exist_ok=True)
You normally do not need to edit this except for:

Changing checkpoint names if your .pth files have different filenames.

Tweaking the default train_mode (e.g., full if you want to finetune encoders as well).

5. Models & Checkpoints
5.1. SAM (ViT-B)
Base checkpoint: weights/sam_vit_b_01ec64.pth

Loaded via sam_froth.models.load_sam_base(...)

Finetune outputs go to:
outputs/sam_finetune_out/

During training we save:

Decoder-only checkpoints (small):
sam_vit_b_<train_mode>_decoder_best.pth, *_decoder_last.pth

Full model checkpoints (big):
sam_vit_b_<train_mode>_full_best.pth, *_full_last.pth

5.2. HQ-SAM (ViT-B)
Base checkpoint: weights/sam_hq_vit_b.pth

Loaded via sam_froth.models.load_hqsam(...)

Finetune outputs go to:
outputs/hqsam_finetune_out/

Same structure for decoder-only and full checkpoints as SAM.

5.3. MedSAM (planned)
The repo is structured so MedSAM can be added later:

Extend sam_froth/models/sam_base.py (or create a dedicated medsam.py).

Add --model medsam routing to scripts/train.py, eval.py, predict.py, amg_demo.py.

Add a MedSAM checkpoint under weights/.

6. Training
All training is driven by:

bash
Copy code
python -m scripts.train --model <sam|hqsam>
6.1. Train SAM
bash
Copy code
# From repo root
python -m scripts.train --model sam
What it does:

Uses FrothSegmentationDataset on data/train.

Uses full-image box prompt [0, 0, W-1, H-1] (semantic-style).

Preprocesses images to 1024×1024 canvas with padding.

Encodes image once with SAM’s image encoder.

Uses BCEDiceLoss on the upsampled, postprocessed logits to original image size.

Runs for EPOCHS_FLEX epochs (configured in scripts/train.py) with:

Gradient accumulation (effective batch size = 1 * ACCUM_STEPS)

Optional mixed precision (AMP) on CUDA

Saves:

decoder-only and full checkpoints under outputs/sam_finetune_out/

updates *_best when validation mIoU improves.

6.2. Train HQ-SAM
bash
Copy code
python -m scripts.train --model hqsam
Differences vs SAM:

Uses HQ-SAM image encoder, which returns:
(img_embed, interm_embeddings)

interm_embeddings are passed into the HQ mask decoder.

Decoder call includes HQ-specific arguments:

hq_token_only=True

interm_embeddings=<list/tensors>

Loss and postprocessing are the same: BCEDiceLoss + SAM-style postprocess (256 → 1024 → crop → original).

6.3. Train Modes
The unified training script supports train modes like:

decoder_only

decoder+prompt

encoder_only

prompt_only

full

Each mode:

Sets requires_grad flags per module (image encoder, prompt encoder, mask decoder).

Builds optimizer param groups with different learning rates per module.

You select via config (in config.py) by changing:

python
Copy code
train_mode = "decoder_only"
or by implementing a CLI argument in scripts/train.py (already set up for internal use).

7. Evaluation
Use:

bash
Copy code
python -m scripts.eval --model sam   --thr 0.5
python -m scripts.eval --model hqsam --thr 0.5
7.1. Split Selection
Eval script chooses split in this priority:

data/eval/ (if exists and non-empty)

data/test/

data/train/

7.2. What eval.py does
For each sample:

Loads finetuned full model:

SAM: best full checkpoint from outputs/sam_finetune_out/

HQ-SAM: best full checkpoint from outputs/hqsam_finetune_out/

Preprocesses image to 1024×1024.

Runs full-image box prompt → one froth mask.

Postprocesses logits to original resolution (H×W) via:

upsample to 1024×1024

crop to resized window

resize to original (H, W)

Applies sigmoid and thresholds at --thr (default 0.5) to produce a binary mask.

Computes per-image:

IoU (Jaccard)

Dice (F1 for binary segmentation)

Prints global:

mIoU

mDice

This gives you semantic segmentation quality of a single unified froth mask, not instance-level segmentation.

8. Prediction (Export Soft Masks)
Use:

bash
Copy code
python -m scripts.predict --model sam
python -m scripts.predict --model hqsam
8.1. Split Selection
Same priority as eval:

data/eval/

data/test/

data/train/

8.2. Output
For each image:

Runs forward pass (same as eval).

Saves soft probability masks (not binary) as grayscale PNGs.

Output path:

text
Copy code
outputs/pred_masks/<model_tag>/soft/mask_XXXX.png
Where:

<model_tag> = e.g. sam_vit_b_decoder_only or hqsam_vit_b_decoder_only

mask_0000.png, mask_0001.png, ...

Pixel values ~ [0, 255] correspond to probability [0, 1].
You can later:

Threshold them in a separate script

Feed into your own post-processing / watershed / froth-counting pipeline.

9. AMG-Style Visualization (Colored Froths)
Use:

bash
Copy code
# SAM AMG
python -m scripts.amg_demo --model sam   --split test --idx 0

# HQ-SAM AMG
python -m scripts.amg_demo --model hqsam --split test --idx 0
Arguments:

--model – sam or hqsam

--split – auto (default), train, test, or eval

--idx – integer index of the sample in the chosen split

9.1. SAM AMG (SamAutomaticMaskGenerator)
For --model sam:

Loads finetuned SAM full model.

Uses SamAutomaticMaskGenerator with tuned parameters:

points_per_side

pred_iou_thresh

stability_score_thresh

min_mask_region_area

Gets a list of masks; for each:

Extracts the largest contour.

Filters tiny blobs by area.

Draws colored contours (different color per froth) on top of the original image.

Displays:

Original image

Ground truth mask

Colored froth contours + counts

9.2. HQ-SAM AMG (Custom Generator)
For --model hqsam:

Loads finetuned HQ-SAM full model.

Uses a custom AMG-style function:

Places a grid of point prompts in 1024-space.

For each point:

Encodes with HQ-SAM (image embeddings + intermediate features).

Decodes a mask.

Postprocesses to original size.

Thresholds by:

mask probability (prob_thr)

predicted IoU (pred_iou_thr)

min area (min_mask_area)

Keeps masks passing thresholds.

Applies NMS by mask IoU (nms_iou_thr) to avoid duplicates.

Visualizes colored contours the same way as SAM AMG.

This gives you per-froth contours derived from HQ-SAM, similar to what you had before but in a clean script.

10. Extending to MedSAM (Roadmap)
The codebase is set up so that adding MedSAM follows the same pattern:

Add a MedSAM loader in sam_froth/models:

load_medsam(checkpoint_path, device, ...)

Add "medsam" as a valid choice:

In scripts/train.py, eval.py, predict.py, amg_demo.py

Add a MedSAM base checkpoint under weights/ and link in config.py:

python
Copy code
medsam_checkpoint = weights_root / "medsam_vit_b.pth"
Reuse:

FrothSegmentationDataset

BCEDiceLoss

postprocess + IoU/Dice

Once wired, the commands would look like:

bash
Copy code
python -m scripts.train   --model medsam
python -m scripts.eval    --model medsam --thr 0.5
python -m scripts.predict --model medsam
python -m scripts.amg_demo --model medsam --idx 0
11. Reproducibility & Tips
Seed: Global seed is set via set_seed(C.seed) (default 1337).

Device:

C.device = "auto" picks CUDA if available, else CPU.

You can force CPU or CUDA in config.py.

Batch size:

DataLoader batch_size is fixed to 1 on purpose for SAM-style prompts.

Effective batch size is increased via gradient accumulation (ACCUM_STEPS in scripts/train.py).

Threshold tuning:

For evaluation, change --thr in:

bash
Copy code
python -m scripts.eval --model sam   --thr 0.5
python -m scripts.eval --model hqsam --thr 0.5
For AMG, tweak:

prob_thr, pred_iou_thr, min_mask_area, nms_iou_thr inside hq_amg_generate or the SAM AMG params inside amg_demo.py.

12. Credits
Authors:

Sina Lotfi

Reza Dadbin

Base models:

SAM: “Segment Anything”, Meta AI

HQ-SAM: “HQ-SAM: Improving the Quality of SAM’s Masks”

MedSAM (planned): “MedSAM: Segment Anything in Medical Images”

This repo just wraps these models into a clean, froth-specific training & evaluation pipeline with a standard Python package + script structure, ready for real-world experiments and future extensions.