import torch


def iou_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute IoU for binary masks.

    pred, target: (H, W) or (1, H, W) or (B, H, W) tensors with {0,1} or {0.,1.}
    Returns a Python float (mean over batch if B > 1).
    """
    if pred.dim() == 3 and pred.size(0) == 1:
        pred = pred[0]
    if target.dim() == 3 and target.size(0) == 1:
        target = target[0]

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    preds = pred.float()
    gts = target.float()

    inter = (preds * gts).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + gts.sum(dim=(1, 2)) - inter + eps
    iou = (inter + eps) / union
    return float(iou.mean().item())
