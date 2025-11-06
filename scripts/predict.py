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


def build_loader(device: torch.device):
    ds = FrothSegmentationDataset(C.test_dir, label_key=C.label_key)
    pin = C.pin_memory and (device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
        pin_memory=pin,
        collate_fn=collate_froth,
    )
    print(f"Predicting on {len(ds)} samples from: {C.test_dir}")
    return loader


def load_finetuned_model(device: torch.device):
    if C.model_name.lower() != "sam":
        raise NotImplementedError("predict.py currently implemented only for model_name='sam'.")

    model = load_sam_base(
        checkpoint_path=C.sam_checkpoint,
        model_type=C.sam_model_type,
        device=device,
    )

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
    return model, model_tag


def main():
    C.setup()
    set_seed(C.seed)
    device = get_device()
    print(f"Using device: {device}")

    loader = build_loader(device)
    model, model_tag = load_finetuned_model(device)

    # save ONLY soft probability maps
    out_dir_soft = C.outputs_root / "pred_masks" / model_tag / "soft"
    out_dir_soft.mkdir(parents=True, exist_ok=True)

    idx_global = 0

    for batch in loader:
        images_1024 = batch["images_1024"].to(device)
        boxes = batch["boxes"].to(device)
        orig_hw = batch["orig_sizes"].to(device)
        input_hw = batch["input_sizes"].to(device)

        with torch.no_grad():
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
            probs = torch.sigmoid(up_logits)  # (B,1,H,W)

        import cv2

        B = probs.shape[0]
        for i in range(B):
            prob = probs[i, 0].cpu().numpy()      # (H,W), float

            # normalize prob to 0..255 for visualization / saving
            p_min, p_max = prob.min(), prob.max()
            if p_max > p_min:
                prob_norm = (prob - p_min) / (p_max - p_min)
            else:
                prob_norm = prob * 0.0
            prob_img = (prob_norm * 255.0).astype(np.uint8)

            fname = f"mask_{idx_global:04d}.png"
            soft_path = out_dir_soft / fname

            cv2.imwrite(str(soft_path), prob_img)
            print(f"Saved soft mask: {soft_path.name}")
            idx_global += 1

    print("\nPrediction complete.")
    print(f"Soft masks saved to: {out_dir_soft}")


if __name__ == "__main__":
    main()
