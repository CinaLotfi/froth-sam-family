import argparse
import random

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset
from sam_froth.models import (
    load_sam_base,
    load_hqsam,
    postprocess_masks_to_original,
)
from sam_froth.models.hqsam import encode_image_hq
from sam_froth.data.froth_dataset import preprocess_image_for_sam  # we’ll reuse 1024-preprocess


# --------- utils ---------
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


def pick_dataset(split: str):
    """
    split: "auto" | "train" | "test" | "eval"
    """
    if split == "eval":
        root = C.eval_dir
    elif split == "test":
        root = C.test_dir
    elif split == "train":
        root = C.train_dir
    else:
        # auto priority
        if C.eval_dir.exists() and any(C.eval_dir.iterdir()):
            root = C.eval_dir
        elif C.test_dir.exists() and any(C.test_dir.iterdir()):
            root = C.test_dir
        else:
            root = C.train_dir

    ds = FrothSegmentationDataset(root, label_key=C.label_key)
    print(f"Using dataset: {root} ({len(ds)} samples)")
    return ds


# --------- SAM AMG (uses official SamAutomaticMaskGenerator) ---------
def amg_sam(device: torch.device, idx: int, split: str):
    from segment_anything import SamAutomaticMaskGenerator

    ds = pick_dataset(split)
    assert 0 <= idx < len(ds), f"Index {idx} out of range (0..{len(ds)-1})"

    image_raw, gt_mask = ds[idx]  # HWC uint8, HxW uint8

    # load finetuned SAM full model
    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM finetuned checkpoint not found at:\n  {ckpt_path}\n"
            "Run: python -m scripts.train --model sam"
        )

    sam = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )
    state = torch.load(ckpt_path, map_location=device)
    sam.load_state_dict(state["model"], strict=True)
    sam.eval()

    # AMG generator
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=80,
    )

    masks = generator.generate(image_raw)
    print(f"[SAM AMG] produced {len(masks)} masks")

    # draw colored contours
    overlay = image_raw.copy()
    union = np.zeros(image_raw.shape[:2], np.uint8)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    n_contours = 0
    for i, m in enumerate(masks):
        seg = (m["segmentation"].astype(np.uint8) * 255)
        union |= seg

        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            continue

        color = colors[i % len(colors)]
        cv2.drawContours(overlay, [cnt], -1, color, 1, lineType=cv2.LINE_AA)
        n_contours += 1

    print("Contours drawn:", n_contours)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.title(f"Original #{idx}"); plt.imshow(image_raw); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("GT mask"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(f"SAM AMG (n_masks={len(masks)}, contours={n_contours})")
    plt.imshow(overlay); plt.axis("off")
    plt.tight_layout(); plt.show()


# --------- HQ-SAM AMG (custom generator) ---------
def hq_amg_generate(
    image_raw_hwc_uint8,
    model,
    device: torch.device,
    points_per_side=32,
    prob_thr=0.5,
    pred_iou_thr=0.80,
    min_mask_area=60,
    nms_iou_thr=0.4,
    hq_token_only=True,
):
    """
    Return list[dict] like AMG:
      { 'segmentation': HxW bool, 'area': int, 'score': float }
    """
    H, W = image_raw_hwc_uint8.shape[:2]

    # preprocess to 1024 canvas (reuse SAM preprocess)
    img_1024, (h_in, w_in) = preprocess_image_for_sam(image_raw_hwc_uint8)
    img_1024 = img_1024.unsqueeze(0).to(device)

    # encode once (HQ needs interm_embeddings)
    with torch.no_grad():
        img_embed, interm_embeddings = encode_image_hq(model, img_1024)

    # grid of points in 1024 space
    xs = torch.linspace(0, max(1, w_in - 1), steps=points_per_side, device=device)
    ys = torch.linspace(0, max(1, h_in - 1), steps=points_per_side, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pts_1024 = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (N,2)

    cand_masks = []
    pe = model.prompt_encoder.get_dense_pe()

    for k in range(pts_1024.shape[0]):
        pt = pts_1024[k:k+1]  # (1,2)
        pts = pt.view(1, 1, 2)
        labs = torch.ones((1, 1), device=device, dtype=torch.int64)  # 1=foreground

        with torch.no_grad():
            sparse_emb, dense_emb = model.prompt_encoder(
                points=(pts, labs),
                boxes=None,
                masks=None,
            )
            lowres_logits, iou_pred = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                hq_token_only=hq_token_only,
                interm_embeddings=interm_embeddings,
            )

            input_hw = torch.tensor([[h_in, w_in]], dtype=torch.int64, device=device)
            orig_hw = torch.tensor([[H, W]], dtype=torch.int64, device=device)
            up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)

            prob = torch.sigmoid(up_logits)[0, 0].detach().cpu().numpy()
            binm = (prob > prob_thr).astype(np.uint8)

        area = int(binm.sum())
        score = float(iou_pred[0, 0].item())
        if area >= min_mask_area and score >= pred_iou_thr:
            cand_masks.append(
                {
                    "segmentation": binm.astype(bool),
                    "area": area,
                    "score": score,
                }
            )

    # simple NMS on masks
    cand_masks.sort(key=lambda m: m["score"], reverse=True)
    keep = []
    for m in cand_masks:
        seg = m["segmentation"]
        ok = True
        for kept in keep:
            inter = np.logical_and(seg, kept["segmentation"]).sum()
            union = seg.sum() + kept["segmentation"].sum() - inter + 1e-6
            iou = inter / union
            if iou >= nms_iou_thr:
                ok = False
                break
        if ok:
            keep.append(m)

    return keep


def amg_hqsam(device: torch.device, idx: int, split: str):
    ds = pick_dataset(split)
    assert 0 <= idx < len(ds), f"Index {idx} out of range (0..{len(ds)-1})"

    image_raw, gt_mask = ds[idx]

    # load finetuned HQ-SAM
    model_tag = f"{C.model_name}_{C.hqsam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HQ-SAM finetuned checkpoint not found at:\n  {ckpt_path}\n"
            "Run: python -m scripts.train --model hqsam"
        )

    sam_hq = load_hqsam(
        checkpoint_path=C.hqsam_checkpoint,
        model_type=C.hqsam_model_type,
        device=device,
    )
    state = torch.load(ckpt_path, map_location=device)
    sam_hq.load_state_dict(state["model"], strict=True)
    sam_hq.eval()

    masks = hq_amg_generate(
        image_raw,
        sam_hq,
        device=device,
        points_per_side=32,
        prob_thr=0.5,
        pred_iou_thr=0.80,
        min_mask_area=60,
        nms_iou_thr=0.32,
    )
    print(f"[HQ-SAM AMG] produced {len(masks)} masks")

    overlay = image_raw.copy()
    union = np.zeros(image_raw.shape[:2], np.uint8)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    n_contours = 0
    for i, m in enumerate(masks):
        seg = (m["segmentation"].astype(np.uint8) * 255)
        union |= seg

        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 10:
            continue

        color = colors[i % len(colors)]
        cv2.drawContours(overlay, [cnt], -1, color, 1, lineType=cv2.LINE_AA)
        n_contours += 1

    print("Contours drawn:", n_contours)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1); plt.title(f"Original #{idx}"); plt.imshow(image_raw); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("GT mask"); plt.imshow(gt_mask, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(f"HQ-SAM AMG (n_masks={len(masks)}, contours={n_contours})")
    plt.imshow(overlay); plt.axis("off")
    plt.tight_layout(); plt.show()


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="AMG-style visualization for SAM / HQ-SAM.")
    p.add_argument(
        "--model",
        type=str,
        default="sam",
        choices=["sam", "hqsam"],
        help="Which model backend to use.",
    )
    p.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Sample index in the chosen split.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "train", "test", "eval"],
        help="Which split to draw from (default: auto priority eval > test > train).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    C.model_name = args.model.lower()
    C.setup()
    set_seed(C.seed)
    device = get_device()

    print(f"Using model: {C.model_name} | device: {device}")

    if args.model == "sam":
        amg_sam(device, idx=args.idx, split=args.split)
    elif args.model == "hqsam":
        amg_hqsam(device, idx=args.idx, split=args.split)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
