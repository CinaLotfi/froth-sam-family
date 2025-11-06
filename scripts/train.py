import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset, collate_froth
from sam_froth.utils.losses import BCEDiceLoss
from sam_froth.models import (
    load_sam_base,
    load_hqsam,
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


def build_loaders(device: torch.device):
    train_ds = FrothSegmentationDataset(C.train_dir, label_key=C.label_key)
    test_ds = FrothSegmentationDataset(C.test_dir, label_key=C.label_key)

    pin = C.pin_memory and (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=C.batch_size,
        shuffle=True,
        num_workers=C.num_workers,
        pin_memory=pin,
        collate_fn=collate_froth,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=pin,
        collate_fn=collate_froth,
    )

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    return train_loader, test_loader


# --------- SAM training ---------
def train_one_epoch_sam(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets = torch.stack(batch["masks"], dim=0).unsqueeze(1).to(device)  # (B,1,H,W)

        # encoder (frozen → no grad)
        with torch.no_grad():
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

        # postprocess + loss
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        loss = criterion(up_logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_epoch_sam(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    ious = []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets_l = batch["masks"]

        img_embed = model.image_encoder(images_1024)
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )
        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)
        preds = (probs > 0.5).float()  # (B,1,H,W)

        B = preds.shape[0]
        for i in range(B):
            gt = targets_l[i].to(device).float()
            p = preds[i, 0]
            inter = (p * gt).sum()
            union = p.sum() + gt.sum() - inter + 1e-6
            iou = (inter + 1e-6) / union
            ious.append(iou.item())

    return float(np.mean(ious)) if ious else 0.0


def train_sam(device: torch.device):
    print("=== Training SAM ===")
    train_loader, test_loader = build_loaders(device)

    # model, decoder-only training
    model = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.mask_decoder.parameters() if p.requires_grad],
        lr=C.lr,
        weight_decay=C.weight_decay,
    )
    criterion = BCEDiceLoss(bce_weight=0.5)

    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    best_decoder_path = C.finetune_out / f"{model_tag}_decoder_best.pth"
    last_decoder_path = C.finetune_out / f"{model_tag}_decoder_last.pth"
    best_full_path = C.finetune_out / f"{model_tag}_full_best.pth"
    last_full_path = C.finetune_out / f"{model_tag}_full_last.pth"

    best_iou = -1.0

    print(
        f"Saving checkpoints under: {C.finetune_out} "
        f"(tag={model_tag})"
    )

    for epoch in range(1, C.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch_sam(model, train_loader, optimizer, criterion, device)
        val_iou = evaluate_epoch_sam(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"[SAM] Epoch {epoch:02d}/{C.epochs} | "
            f"loss={tr_loss:.4f} | val mIoU={val_iou:.4f} | {dt:.1f}s"
        )

        # save last
        torch.save(
            {
                "mask_decoder": model.mask_decoder.state_dict(),
                "model_type": C.sam_model_type,
                "epoch": epoch,
                "val_mIoU": val_iou,
            },
            last_decoder_path,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": C.sam_model_type,
                "epoch": epoch,
                "val_mIoU": val_iou,
            },
            last_full_path,
        )

        # save best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "mask_decoder": model.mask_decoder.state_dict(),
                    "model_type": C.sam_model_type,
                    "epoch": epoch,
                    "val_mIoU": val_iou,
                },
                best_decoder_path,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": C.sam_model_type,
                    "epoch": epoch,
                    "val_mIoU": val_iou,
                },
                best_full_path,
            )
            print(
                f"  ✅ New best mIoU {best_iou:.4f} saved:\n"
                f"     - decoder: {best_decoder_path.name}\n"
                f"     - full   : {best_full_path.name}"
            )

    print("\nSAM training done.")
    print(f"Best mIoU: {best_iou:.4f}")


# --------- HQ-SAM training ---------
def train_one_epoch_hq(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    hq_token_only: bool = True,
) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets = torch.stack(batch["masks"], dim=0).unsqueeze(1).to(device)  # (B,1,H,W)

        # HQ encoder (decoder-only setup → no grad here)
        with torch.no_grad():
            img_embed, interm_embeddings = encode_image_hq(model, images_1024)

        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
        )

        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        loss = criterion(up_logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_epoch_hq(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hq_token_only: bool = True,
) -> float:
    model.eval()
    ious = []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)
        targets_l = batch["masks"]

        img_embed, interm_embeddings = encode_image_hq(model, images_1024)

        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        lowres_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
        )

        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)
        preds = (probs > 0.5).float()  # (B,1,H,W)

        B = preds.shape[0]
        for i in range(B):
            gt = targets_l[i].to(device).float()
            p = preds[i, 0]
            inter = (p * gt).sum()
            union = p.sum() + gt.sum() - inter + 1e-6
            iou = (inter + 1e-6) / union
            ious.append(iou.item())

    return float(np.mean(ious)) if ious else 0.0


def train_hqsam(device: torch.device):
    print("=== Training HQ-SAM ===")
    train_loader, test_loader = build_loaders(device)

    model = load_hqsam(
        checkpoint_path=C.hqsam_checkpoint,
        model_type=C.hqsam_model_type,
        device=device,
    )

    # decoder-only training
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.mask_decoder.parameters() if p.requires_grad],
        lr=C.lr,
        weight_decay=C.weight_decay,
    )
    criterion = BCEDiceLoss(bce_weight=0.5)

    model_tag = f"{C.model_name}_{C.hqsam_model_type}_{C.train_mode}"
    best_decoder_path = C.finetune_out / f"{model_tag}_decoder_best.pth"
    last_decoder_path = C.finetune_out / f"{model_tag}_decoder_last.pth"
    best_full_path = C.finetune_out / f"{model_tag}_full_best.pth"
    last_full_path = C.finetune_out / f"{model_tag}_full_last.pth"

    best_iou = -1.0

    print(
        f"Saving checkpoints under: {C.finetune_out} "
        f"(tag={model_tag})"
    )

    for epoch in range(1, C.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch_hq(model, train_loader, optimizer, criterion, device)
        val_iou = evaluate_epoch_hq(model, test_loader, device)
        dt = time.time() - t0

        print(
            f"[HQ-SAM] Epoch {epoch:02d}/{C.epochs} | "
            f"loss={tr_loss:.4f} | val mIoU={val_iou:.4f} | {dt:.1f}s"
        )

        torch.save(
            {
                "mask_decoder": model.mask_decoder.state_dict(),
                "model_type": C.hqsam_model_type,
                "epoch": epoch,
                "val_mIoU": val_iou,
            },
            last_decoder_path,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": C.hqsam_model_type,
                "epoch": epoch,
                "val_mIoU": val_iou,
            },
            last_full_path,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "mask_decoder": model.mask_decoder.state_dict(),
                    "model_type": C.hqsam_model_type,
                    "epoch": epoch,
                    "val_mIoU": val_iou,
                },
                best_decoder_path,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_type": C.hqsam_model_type,
                    "epoch": epoch,
                    "val_mIoU": val_iou,
                },
                best_full_path,
            )
            print(
                f"  ✅ New best mIoU {best_iou:.4f} saved:\n"
                f"     - decoder: {best_decoder_path.name}\n"
                f"     - full   : {best_full_path.name}"
            )

    print("\nHQ-SAM training done.")
    print(f"Best mIoU: {best_iou:.4f}")


# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser(description="Train froth SAM-family models.")
    p.add_argument(
        "--model",
        type=str,
        default="sam",
        choices=["sam", "hqsam"],
        help="Which model backend to train.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # tell config which model we're using
    C.model_name = args.model.lower()
    C.setup()
    set_seed(C.seed)
    device = get_device()

    print(f"Using model: {C.model_name} | device: {device}")
    print(f"Train dir: {C.train_dir}")
    print(f"Test dir : {C.test_dir}")

    if args.model == "sam":
        train_sam(device)
    elif args.model == "hqsam":
        train_hqsam(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
