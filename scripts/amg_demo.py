import random
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if C.device == "cuda":
        return torch.device("cuda")
    if C.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam_for_amg(device: torch.device, use_full: bool = True):
    """
    Load SAM for AMG:
      - if use_full=True: load full finetuned model checkpoint
      - if use_full=False: load base SAM + finetuned decoder-only checkpoint
    """
    if C.model_name.lower() != "sam":
        raise NotImplementedError("AMG demo currently implemented only for model_name='sam'.")

    base_ckpt = C.sam_checkpoint
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Base SAM checkpoint not found at {base_ckpt}. "
            "Download sam_vit_b_01ec64.pth into weights/."
        )

    sam = sam_model_registry[C.sam_model_type](checkpoint=str(base_ckpt)).to(device).eval()

    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    full_ckpt = C.finetune_out / f"{model_tag}_full_best.pth"
    dec_ckpt  = C.finetune_out / f"{model_tag}_decoder_best.pth"

    if use_full:
        if not full_ckpt.exists():
            raise FileNotFoundError(
                f"Full finetuned checkpoint not found at:\n  {full_ckpt}\n"
                "Run scripts/train.py first or set use_full=False to try decoder-only."
            )
        state = torch.load(full_ckpt, map_location=device)
        sam.load_state_dict(state["model"], strict=True)
        print(f"Loaded FULL finetuned model: {full_ckpt.name}")
        print(f"  epoch={state.get('epoch')}, val_mIoU={state.get('val_mIoU'):.4f}")
    else:
        if not dec_ckpt.exists():
            raise FileNotFoundError(
                f"Decoder-only checkpoint not found at:\n  {dec_ckpt}\n"
                "Run scripts/train.py first or set use_full=True to load full model."
            )
        state = torch.load(dec_ckpt, map_location=device)
        sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        print(f"Loaded base SAM + finetuned DECODER: {dec_ckpt.name}")

    return sam, model_tag


def main():
    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    # load model for AMG (set use_full=False if you prefer base+decoder-only)
    USE_FULL = True
    sam, model_tag = load_sam_for_amg(device, use_full=USE_FULL)

    # dataset (prefer test set if available)
    if C.test_dir.exists() and any(C.test_dir.iterdir()):
        ds_root = C.test_dir
        print(f"Using test set for AMG: {ds_root}")
    else:
        ds_root = C.train_dir
        print(f"[WARN] Test dir empty/not found, using train dir: {ds_root}")

    ds = FrothSegmentationDataset(ds_root, label_key=C.label_key)

    # pick which sample to visualize
    idx = 0  # change if you want another image
    image_raw, gt_mask = ds[idx]   # HWC uint8, HxW uint8
    print(f"Running AMG on sample #{idx}…")

    # AMG generator (tune thresholds to your liking)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=80,
    )

    masks = generator.generate(image_raw)
    print(f"AMG produced {len(masks)} masks")

    # ---------- colored froths overlay ----------
    overlay = image_raw.copy()
    h, w = overlay.shape[:2]

    # accumulate colored froth regions here
    color_layer = np.zeros_like(overlay, dtype=np.uint8)
    n_kept = 0

    # some RGB colors (cycled through froths)
    colors_rgb = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
    ]

    for m in masks:
        seg = (m["segmentation"].astype(np.uint8) * 255)

        # find main contour of this mask
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            continue

        # pick a color in BGR for cv2
        r, g, b = colors_rgb[n_kept % len(colors_rgb)]
        color_bgr = (b, g, r)
        n_kept += 1

        # fill this froth region on the color layer
        cv2.drawContours(color_layer, [cnt], -1, color_bgr, thickness=-1)

    # alpha blend colored masks on top of original image
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_layer, alpha, 0)

    print("Number of colored froth regions:", n_kept)

    # ---------- visualize ----------
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Original #{idx}")
    plt.imshow(image_raw)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground truth mask")
    plt.imshow(gt_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"AMG froths (regions={n_kept}, masks={len(masks)})")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
