from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
except Exception as e:
    raise RuntimeError(
        "segment-anything package not found.\n"
        "Install with:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


TrainMode = Literal["decoder_only", "decoder+prompt", "prompt_only", "encoder_only", "full"]


def load_sam_base(
    checkpoint_path: Path | str,
    model_type: str,
    device: torch.device,
) -> Sam:
    """
    Load a base SAM model (vit_b/l/h) with pretrained checkpoint and move to device.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at: {checkpoint_path}\n"
            "Place the correct .pth file in the weights/ directory."
        )

    model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    model.to(device)
    return model


def set_trainables(model: Sam, mode: TrainMode) -> None:
    """
    Freeze/unfreeze SAM submodules according to training mode.
    """
    mode = mode.lower()

    # default: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # decoder
    if mode in {"decoder_only", "decoder+prompt", "full"}:
        for p in model.mask_decoder.parameters():
            p.requires_grad = True

    # prompt encoder
    if mode in {"prompt_only", "decoder+prompt", "full"}:
        for p in model.prompt_encoder.parameters():
            p.requires_grad = True

    # image encoder
    if mode in {"encoder_only", "full"}:
        for p in model.image_encoder.parameters():
            p.requires_grad = True


def count_trainable_params(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def boxes_to_1024_space(boxes: torch.Tensor, orig_hw: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from ORIGINAL pixel coords to SAM's 1024×1024 prompt space.

    boxes:   (B, 4)  [x0, y0, x1, y1] in original image coords
    orig_hw: (B, 2)  [H, W]
    returns: (B, 4)  scaled into 1024-space
    """
    out = []
    device = boxes.device
    for i in range(boxes.shape[0]):
        H, W = orig_hw[i].tolist()
        sx = 1024.0 / float(W)
        sy = 1024.0 / float(H)
        x0, y0, x1, y1 = boxes[i].tolist()
        out.append([x0 * sx, y0 * sy, x1 * sx, y1 * sy])
    return torch.tensor(out, dtype=torch.float32, device=device)


def postprocess_masks_to_original(
    logits_lowres: torch.Tensor,
    input_hw: torch.Tensor,   # (B,2): (h_resized, w_resized) BEFORE padding
    orig_hw: torch.Tensor,    # (B,2): (H, W) original
) -> torch.Tensor:
    """
    SAM-compatible postprocess:

      1) (B,1,256,256) -> upsample to (B,1,1024,1024)
      2) crop to the resized window (h_resized, w_resized) BEFORE padding
      3) resize to original (H, W)

    Returns: (B,1,H,W) logits with gradients.
    """
    outs = []
    B = logits_lowres.shape[0]
    for i in range(B):
        h_in, w_in = map(int, input_hw[i].tolist())
        H, W = map(int, orig_hw[i].tolist())

        # 1) upsample to 1024x1024
        up_1024 = F.interpolate(
            logits_lowres[i:i + 1], size=(1024, 1024),
            mode="bilinear", align_corners=False
        )
        # 2) crop away zero padding
        up_cropped = up_1024[..., :h_in, :w_in]
        # 3) resize to original size
        up_orig = F.interpolate(
            up_cropped, size=(H, W),
            mode="bilinear", align_corners=False
        )
        outs.append(up_orig)

    return torch.cat(outs, dim=0)  # (B,1,H,W)
