# sam_froth/models/medsam.py

from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from segment_anything import sam_model_registry


def load_medsam(
    checkpoint_path: Path,
    model_type: str = "vit_b",
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Load MedSAM (ViT-B) using the same registry as SAM,
    but with your MedSAM checkpoint.

    Tries:
      1) sam_model_registry[model_type](checkpoint=...)
      2) manual state_dict load with strict=False
    """
    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[MedSAM] Checkpoint not found: {checkpoint_path}"
        )

    # ---- 1) try the native way ----
    try:
        model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        model.to(device)
        print(f"[MedSAM] Loaded via registry: {checkpoint_path.name}")
        return model
    except Exception as e:
        print(f"[MedSAM] registry(checkpoint=...) failed: {e}")
        print("[MedSAM] Falling back to manual state_dict loading...")

    # ---- 2) fallback: manual state_dict load ----
    model = sam_model_registry[model_type]()  # init without weights

    sd: Dict[str, Any] = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        state = sd["model"]
    else:
        state = sd  # assume raw state_dict

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[MedSAM] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[MedSAM] Unexpected keys: {len(unexpected)}")

    model.to(device)
    print(f"[MedSAM] Loaded with non-strict state_dict from {checkpoint_path.name}")
    return model
