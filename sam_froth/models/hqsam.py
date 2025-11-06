import torch
import torch.nn as nn

from segment_anything_hq import sam_model_registry as hq_sam_model_registry


def load_hqsam(checkpoint_path, model_type: str, device: torch.device) -> nn.Module:
    """
    Load an HQ-SAM model (vit_b / vit_l / vit_h) from checkpoint.
    """
    checkpoint_path = str(checkpoint_path)
    model = hq_sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    model.eval()
    return model


def encode_image_hq(model: nn.Module, images_1024: torch.Tensor):
    """
    Run HQ-SAM image encoder and return:
      img_embed, interm_embeddings

    Handles different variants of segment-anything-hq that may return:
      - (img_embed, interm_embeddings) tuple
      - just img_embed + attributes on encoder / predictor
    """
    enc_out = model.image_encoder(images_1024)

    # Newer versions: (embed, interm_embeddings)
    if isinstance(enc_out, tuple) and len(enc_out) == 2:
        img_embed, interm_embeddings = enc_out
    else:
        # Fallbacks seen in some builds
        if hasattr(model, "predictor") and hasattr(model.predictor, "interm_features"):
            img_embed = model.predictor.get_image_embedding()
            interm_embeddings = model.predictor.interm_features
        elif hasattr(model.image_encoder, "interm_embeddings"):
            img_embed = enc_out
            interm_embeddings = model.image_encoder.interm_embeddings
        else:
            raise RuntimeError(
                "HQ-SAM encoder did not expose intermediate embeddings.\n"
                "Update 'segment-anything-hq' so image_encoder returns "
                "(img_embed, interm_embeddings)."
            )

    # Ensure proper device / type
    if isinstance(interm_embeddings, list):
        interm_embeddings = [t.to(img_embed.device) for t in interm_embeddings]
    elif torch.is_tensor(interm_embeddings):
        interm_embeddings = interm_embeddings.to(img_embed.device)
    else:
        raise TypeError(
            "Unexpected type for interm_embeddings; expected list[Tensor] or Tensor."
        )

    return img_embed, interm_embeddings
