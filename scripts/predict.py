import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset, collate_froth
from sam_froth.models import (
    load_sam_base,
    load_hqsam,
    load_medsam,              # 👈 NEW
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
    For prediction we prefer:
      - eval_dir if non-empty
      - else test_dir
      - else train_dir
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
    print(f"Predicting on {len(ds)} samples from [{split_name}] dir: {root}")
    return loader


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
    return model, model_tag


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
    return model, model_tag


def load_finetuned_medsam(device: torch.device):
    """
    MedSAM loader:
      - start from base MedSAM weights
      - optionally load finetuned full_best checkpoint (if present & valid)
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
        print("[MedSAM] No finetuned checkpoint found, using base MedSAM weights only.")
        model.eval()
        return model, model_tag

    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MedSAM finetuned checkpoint at:\n  {ckpt_path}\n"
            f"Delete it and re-run training:\n"
            f"  python -m scripts.train --model medsam\n"
            f"Original torch.load error:\n  {e}"
        ) from e

    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()
    print(f"Loaded MedSAM finetuned model: {ckpt_path.name}")
    return model, model_tag


# --------- prediction backends (soft maps only) ---------
@torch.no_grad()
def predict_sam(device: torch.device):
    loader = build_loader()
    model, model_tag = load_finetuned_sam(device)

    out_dir_soft = C.outputs_root / "pred_masks" / model_tag / "soft"
    out_dir_soft.mkdir(parents=True, exist_ok=True)
    print(f"Saving SAM soft masks to: {out_dir_soft}")

    idx_global = 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

        # encoder
        img_embed = model.image_encoder(images_1024)

        # prompts
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        # decode
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # postprocess to original
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)  # (B,1,H,W)

        B = probs.shape[0]
        for i in range(B):
            prob = probs[i, 0].cpu().numpy()  # (H,W)

            # normalize to 0..255 for saving
            p_min, p_max = prob.min(), prob.max()
            if p_max > p_min:
                prob_norm = (prob - p_min) / (p_max - p_min)
            else:
                prob_norm = prob * 0.0
            prob_img = (prob_norm * 255.0).astype(np.uint8)

            fname = f"mask_{idx_global:04d}.png"
            soft_path = out_dir_soft / fname
            cv2.imwrite(str(soft_path), prob_img)
            print(f"[SAM] saved soft mask: {soft_path.name}")
            idx_global += 1

    print("\nSAM prediction complete.")


@torch.no_grad()
def predict_hqsam(device: torch.device):
    loader = build_loader()
    model, model_tag = load_finetuned_hqsam(device)

    out_dir_soft = C.outputs_root / "pred_masks" / model_tag / "soft"
    out_dir_soft.mkdir(parents=True, exist_ok=True)
    print(f"Saving HQ-SAM soft masks to: {out_dir_soft}")

    idx_global = 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

        # HQ encoder: img_embed + interm_embeddings
        img_embed, interm_embeddings = encode_image_hq(model, images_1024)

        # prompts
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        # HQ decoder (must pass HQ args)
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings,
        )

        # postprocess to original
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)  # (B,1,H,W)

        B = probs.shape[0]
        for i in range(B):
            prob = probs[i, 0].cpu().numpy()  # (H,W)

            # normalize to 0..255 for saving
            p_min, p_max = prob.min(), prob.max()
            if p_max > p_min:
                prob_norm = (prob - p_min) / (p_max - p_min)
            else:
                prob_norm = prob * 0.0
            prob_img = (prob_norm * 255.0).astype(np.uint8)

            fname = f"mask_{idx_global:04d}.png"
            soft_path = out_dir_soft / fname
            cv2.imwrite(str(soft_path), prob_img)
            print(f"[HQ-SAM] saved soft mask: {soft_path.name}")
            idx_global += 1

    print("\nHQ-SAM prediction complete.")


@torch.no_grad()
def predict_medsam(device: torch.device):
    loader = build_loader()
    model, model_tag = load_finetuned_medsam(device)

    out_dir_soft = C.outputs_root / "pred_masks" / model_tag / "soft"
    out_dir_soft.mkdir(parents=True, exist_ok=True)
    print(f"Saving MedSAM soft masks to: {out_dir_soft}")

    idx_global = 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

        # encoder (same API as base SAM)
        img_embed = model.image_encoder(images_1024)

        # prompts: full-image box
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

        # postprocess to original
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)  # (B,1,H,W)

        B = probs.shape[0]
        for i in range(B):
            prob = probs[i, 0].cpu().numpy()  # (H,W)

            # normalize to 0..255 for saving
            p_min, p_max = prob.min(), prob.max()
            if p_max > p_min:
                prob_norm = (prob - p_min) / (p_max - p_min)
            else:
                prob_norm = prob * 0.0
            prob_img = (prob_norm * 255.0).astype(np.uint8)

            fname = f"mask_{idx_global:04d}.png"
            soft_path = out_dir_soft / fname
            cv2.imwrite(str(soft_path), prob_img)
            print(f"[MedSAM] saved soft mask: {soft_path.name}")
            idx_global += 1

    print("\nMedSAM prediction complete.")


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Predict froth masks with SAM-family models.")
    p.add_argument(
        "--model",
        type=str,
        default="sam",
        choices=["sam", "hqsam", "medsam"],   # 👈 NEW
        help="Which model backend to use.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # wire config
    C.model_name = args.model.lower()
    C.setup()
    set_seed(C.seed)
    device = get_device()

    print(f"Using model: {C.model_name} | device: {device}")

    if args.model == "sam":
        predict_sam(device)
    elif args.model == "hqsam":
        predict_hqsam(device)
    elif args.model == "medsam":
        predict_medsam(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
