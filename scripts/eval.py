import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset, collate_froth
from sam_froth.models import (
    load_sam_base,
    load_hqsam,
    load_medsam,                 # 👈 NEW
    boxes_to_1024_space,
    postprocess_masks_to_original,
)
from sam_froth.models.hqsam import encode_image_hq


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


def build_loader():
    """
    Eval priority:
      1) eval_dir  (if exists and non-empty)
      2) test_dir
      3) train_dir
    """
    if C.eval_dir.exists() and any(C.eval_dir.iterdir()):
        root = C.eval_dir
        split_name = "eval"
    elif C.test_dir.exists() and any(C.test_dir.iterdir()):
        root = C.test_dir
        split_name = "test"
    else:
        root = C.train_dir
        split_name = "train"

    ds = FrothSegmentationDataset(root, label_key=C.label_key)
    loader = DataLoader(
        ds,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=(get_device().type == "cuda") and C.pin_memory,
        collate_fn=collate_froth,
    )
    print(f"Evaluating on {len(ds)} samples from [{split_name}] dir: {root}")
    return loader


def compute_iou_and_dice(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6):
    """
    pred_mask, true_mask: (H,W) binary arrays (0/1 or False/True)
    """
    pred = pred_mask.astype(bool)
    target = true_mask.astype(bool)

    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = (inter + eps) / (union + eps)

    dice = (2 * inter + eps) / (pred.sum() + target.sum() + eps)
    return float(iou), float(dice)


# --------- model loading ---------
def load_finetuned_sam(device: torch.device):
    model = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )

    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"SAM finetuned checkpoint not found at:\n  {ckpt_path}\n"
            "Run: python -m scripts.train --model sam"
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    print(f"Loaded SAM finetuned model: {ckpt_path.name}")
    return model


def load_finetuned_hqsam(device: torch.device):
    model = load_hqsam(
        checkpoint_path=C.hqsam_checkpoint,
        model_type=C.hqsam_model_type,
        device=device,
    )

    model_tag = f"{C.model_name}_{C.hqsam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"HQ-SAM finetuned checkpoint not found at:\n  {ckpt_path}\n"
            "Run: python -m scripts.train --model hqsam"
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    print(f"Loaded HQ-SAM finetuned model: {ckpt_path.name}")
    return model


def load_finetuned_medsam(device: torch.device):
    """
    MedSAM: same API as SAM, just different checkpoint + output dir.
    Falls back to base MedSAM if no finetuned ckpt is found.
    """
    model = load_medsam(
        checkpoint_path=C.medsam_checkpoint,
        model_type=C.medsam_model_type,
        device=device,
    )

    model_tag = f"{C.model_name}_{C.medsam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"

    print(f"[MedSAM] Looking for finetuned checkpoint at: {ckpt_path}")

    if not ckpt_path.exists():
        print(
            "[MedSAM] No finetuned checkpoint found, "
            "using base MedSAM weights only."
        )
        model.eval()
        return model

    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MedSAM finetuned checkpoint at:\n  {ckpt_path}\n"
            f"File is likely corrupted. Delete it and re-run training:\n"
            f"  python -m scripts.train --model medsam\n"
            f"Original torch.load error:\n  {e}"
        ) from e

    # Support both { 'model': state_dict } and raw state_dict
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    print(f"Loaded MedSAM finetuned model: {ckpt_path.name}")
    return model


# --------- eval backends ---------
@torch.no_grad()
def eval_sam_like(model, device: torch.device, threshold: float):
    """
    Shared eval for SAM + MedSAM (same forward API).
    """
    loader = build_loader()
    all_ious, all_dices = [], []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets_l = batch["masks"]  # list[(H,W) float {0,1}]

        # encoder
        img_embed = model.image_encoder(images_1024)

        # prompts
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        # decoder
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # postprocess and metrics
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
        probs = torch.sigmoid(up_logits)
        preds = (probs > threshold).float()

        B = preds.shape[0]
        for i in range(B):
            pred_np = preds[i, 0].cpu().numpy()
            gt_np = targets_l[i].cpu().numpy()  # (H,W) 0/1

            iou, dice = compute_iou_and_dice(pred_np, gt_np)
            all_ious.append(iou)
            all_dices.append(dice)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_dice = float(np.mean(all_dices)) if all_dices else 0.0

    return mean_iou, mean_dice


@torch.no_grad()
def eval_sam(device: torch.device, threshold: float):
    model = load_finetuned_sam(device)
    mean_iou, mean_dice = eval_sam_like(model, device, threshold)
    print(f"\n[SAM] Eval @ thr={threshold:.2f} → mIoU={mean_iou:.4f}, mDice={mean_dice:.4f}")
    return mean_iou, mean_dice


@torch.no_grad()
def eval_medsam(device: torch.device, threshold: float):
    model = load_finetuned_medsam(device)
    mean_iou, mean_dice = eval_sam_like(model, device, threshold)
    print(f"\n[MedSAM] Eval @ thr={threshold:.2f} → mIoU={mean_iou:.4f}, mDice={mean_dice:.4f}")
    return mean_iou, mean_dice


@torch.no_grad()
def eval_hqsam(device: torch.device, threshold: float):
    loader = build_loader()
    model = load_finetuned_hqsam(device)

    all_ious = []
    all_dices = []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets_l = batch["masks"]

        # HQ encoder: img_embed + interm_embeddings
        img_embed, interm_embeddings = encode_image_hq(model, images_1024)

        # prompts
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        # HQ decoder
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings,
        )

        # postprocess and metrics
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)  # (B,1,H,W)
        probs = torch.sigmoid(up_logits)
        preds = (probs > threshold).float()

        B = preds.shape[0]
        for i in range(B):
            pred_np = preds[i, 0].cpu().numpy()
            gt_np = targets_l[i].cpu().numpy()

            iou, dice = compute_iou_and_dice(pred_np, gt_np)
            all_ious.append(iou)
            all_dices.append(dice)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_dice = float(np.mean(all_dices)) if all_dices else 0.0

    print(f"\n[HQ-SAM] Eval @ thr={threshold:.2f} → mIoU={mean_iou:.4f}, mDice={mean_dice:.4f}")
    return mean_iou, mean_dice


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate froth SAM-family models.")
    p.add_argument(
        "--model",
        type=str,
        default="sam",
        choices=["sam", "hqsam", "medsam"],   # 👈 NEW
        help="Which model backend to evaluate.",
    )
    p.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Probability threshold to binarize masks.",
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
        eval_sam(device, threshold=args.thr)
    elif args.model == "hqsam":
        eval_hqsam(device, threshold=args.thr)
    elif args.model == "medsam":
        eval_medsam(device, threshold=args.thr)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
