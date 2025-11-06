import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config as C
from sam_froth.data.froth_dataset import FrothSegmentationDataset, collate_froth
from sam_froth.models import (
    load_sam_base,
    boxes_to_1024_space,
    postprocess_masks_to_original,
)
from sam_froth.utils.metrics import iou_binary


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


def build_test_loader(device: torch.device):
    test_ds = FrothSegmentationDataset(C.test_dir, label_key=C.label_key)
    pin = C.pin_memory and (device.type == "cuda")
    test_loader = DataLoader(
        test_ds,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=pin,
        collate_fn=collate_froth,
    )
    print(f"Test samples: {len(test_ds)}")
    return test_loader


def load_finetuned_model(device: torch.device):
    if C.model_name.lower() != "sam":
        raise NotImplementedError("Eval currently implemented only for model_name='sam'.")

    # base SAM
    model = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )

    # expected checkpoint path (same pattern as train.py)
    model_tag = f"{C.model_name}_{C.sam_model_type}_{C.train_mode}"
    ckpt_path = C.finetune_out / f"{model_tag}_full_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Finetuned checkpoint not found at:\n  {ckpt_path}\n"
            "Run scripts/train.py first or adjust the path logic."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    print(f"Loaded finetuned model from: {ckpt_path}")
    print(f"  epoch={state.get('epoch')}, val_mIoU={state.get('val_mIoU'):.4f}")
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ious = []

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

        # encode image
        img_embed = model.image_encoder(images_1024)

        # prompt: full-image box in 1024-space
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

        # postprocess to original size
        up_logits = postprocess_masks_to_original(lowres_logits, input_hw, orig_hw)
        probs = torch.sigmoid(up_logits)
        preds = (probs > 0.5).float()  # (B,1,H,W)

        B = preds.shape[0]
        for i in range(B):
            gt = batch["masks"][i].to(device).float()           # (H,W)
            p = preds[i, 0]                                     # (H,W)
            iou = iou_binary(p, gt)
            ious.append(iou)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou


def main():
    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    test_loader = build_test_loader(device)
    model = load_finetuned_model(device)

    mean_iou = evaluate(model, test_loader, device)
    print(f"\nFinal mean IoU on test set: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
