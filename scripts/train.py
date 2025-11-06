import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset, collate_froth
from sam_froth.utils.losses import BCEDiceLoss
from sam_froth.models import (
    load_sam_base,
    set_trainables,
    count_trainable_params,
    boxes_to_1024_space,
    postprocess_masks_to_original,
)


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
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(device: torch.device):
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


def build_model_and_optimizer(device: torch.device):
    if C.model_name.lower() != "sam":
        raise NotImplementedError(
            f"Only 'sam' is implemented for now. Got model_name={C.model_name!r}"
        )

    # load base SAM
    model = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )

    # set trainable parts
    set_trainables(model, C.train_mode)

    # optional warm-start decoder
    if C.load_decoder_from is not None:
        dec_path = Path(C.load_decoder_from)
        if dec_path.exists():
            state = torch.load(dec_path, map_location=device)
            if "mask_decoder" in state:
                model.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
                print(f"Warm-started mask_decoder from: {dec_path}")
        else:
            print(f"[WARN] load_decoder_from path does not exist: {dec_path}")

    # build param groups
    param_groups = []
    if C.lr_decoder > 0:
        params = [p for p in model.mask_decoder.parameters() if p.requires_grad]
        if params:
            param_groups.append(
                {"params": params, "lr": C.lr_decoder, "weight_decay": C.weight_decay}
            )
    if C.lr_prompt > 0:
        params = [p for p in model.prompt_encoder.parameters() if p.requires_grad]
        if params:
            param_groups.append(
                {"params": params, "lr": C.lr_prompt, "weight_decay": C.weight_decay}
            )
    if C.lr_encoder > 0:
        params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        if params:
            param_groups.append(
                {"params": params, "lr": C.lr_encoder, "weight_decay": C.weight_decay}
            )

    if not any(g["params"] for g in param_groups):
        raise RuntimeError(
            "No trainable parameters. Check train_mode and learning rates in config."
        )

    optimizer = torch.optim.AdamW(param_groups)

    print(f"Trainable params: {count_trainable_params(model):,}")
    return model, optimizer


def train_one_epoch_flex(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    criterion: nn.Module,
    accum_steps: int,
    grad_clip_norm: float | None,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_steps = 0

    need_img_grads = any(p.requires_grad for p in model.image_encoder.parameters())

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images_1024 = batch["images_1024"].to(device)  # (B,3,1024,1024)
        boxes = batch["boxes"].to(device)              # (B,4)
        orig_hw = batch["orig_sizes"].to(device)       # (B,2)
        input_hw = batch["input_sizes"].to(device)     # (B,2)

        # image encoder
        if need_img_grads:
            img_embed = model.image_encoder(images_1024)
        else:
            with torch.no_grad():
                img_embed = model.image_encoder(images_1024)

        # prompt encoder
        boxes_1024 = boxes_to_1024_space(boxes, orig_hw)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=boxes_1024.unsqueeze(1),
            masks=None,
        )

        with autocast(enabled=use_amp):
            lowres_logits, _ = model.mask_decoder(
                image_embeddings=img_embed,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            up_logits = postprocess_masks_to_original(
                lowres_logits, input_hw, orig_hw
            )  # (B,1,H,W)

            # stack masks list into (B,1,H,W)
            masks = [m.to(device) for m in batch["masks"]]
            target = torch.stack(masks, dim=0).unsqueeze(1)  # (B,1,H,W)

            loss = criterion(up_logits, target) / accum_steps

        scaler.scale(loss).backward()

        if ((step + 1) % accum_steps) == 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps
        n_steps += 1

    return running_loss / max(1, n_steps)


@torch.no_grad()
def evaluate_epoch_flex(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    model.eval()
    ious = []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

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
        preds = (probs > threshold).float()  # (B,1,H,W)

        B = preds.shape[0]
        for i in range(B):
            gt = batch["masks"][i].to(device).float()
            p = preds[i, 0]
            inter = (p * gt).sum()
            union = p.sum() + gt.sum() - inter + 1e-6
            iou = (inter + 1e-6) / union
            ious.append(iou.item())

    return float(np.mean(ious)) if ious else 0.0


def main():
    # setup
    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    # data
    train_loader, test_loader = build_dataloaders(device)

    # model + optimizer + loss
    model, optimizer = build_model_and_optimizer(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    use_amp = C.use_amp and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # checkpoint paths
    out_dir = C.finetune_out
    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    best_iou = -1.0

    best_decoder_path = out_dir / f"{model_tag}_decoder_best.pth"
    last_decoder_path = out_dir / f"{model_tag}_decoder_last.pth"
    best_full_path = out_dir / f"{model_tag}_full_best.pth"
    last_full_path = out_dir / f"{model_tag}_full_last.pth"

    print(f"Starting training for {C.epochs} epoch(s) with accumulation x{C.accum_steps}…")

    for epoch in range(1, C.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_flex(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            criterion=criterion,
            accum_steps=C.accum_steps,
            grad_clip_norm=C.grad_clip_norm,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_iou = evaluate_epoch_flex(
            model=model,
            loader=test_loader,
            device=device,
            threshold=0.5,
        )
        dt = time.time() - t0

        print(
            f"[{C.train_mode}] Epoch {epoch:02d}/{C.epochs} | "
            f"loss={train_loss:.4f} | val_mIoU={val_iou:.4f} | {dt:.1f}s"
        )

        # save decoder-only (always)
        torch.save(
            {
                "mask_decoder": model.mask_decoder.state_dict(),
                "model_type": C.sam_model_type,
                "epoch": epoch,
                "val_mIoU": val_iou,
            },
            last_decoder_path,
        )

        # save full model
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": C.sam_model_type,
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
                f"     - decoder: {best_decoder_path}\n"
                f"     - full   : {best_full_path}"
            )

    print("\nDone.")
    print(f"Best mIoU: {best_iou:.4f}")
    print(f"Decoder checkpoints: best={best_decoder_path}, last={last_decoder_path}")
    print(f"Full checkpoints   : best={best_full_path}, last={last_full_path}")


if __name__ == "__main__":
    main()
