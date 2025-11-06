import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined BCE-with-logits + soft Dice loss.

    Expects:
      - logits: Tensor (B, 1, H, W)   (raw outputs from decoder, before sigmoid)
      - target: Tensor (B, 1, H, W)   (float {0,1}, same size as logits)
    """
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE part
        bce = self.bce(logits, target)

        # Soft Dice part
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.size(0), -1)
        target_f = target.view(target.size(0), -1)

        intersection = (probs_f * target_f).sum(dim=1)
        union = probs_f.sum(dim=1) + target_f.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps))
        dice = dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
