import torch
import torch.nn as nn
import torch.nn.functional as F


def get_edge_gt(mask: torch.Tensor, dilation: int = 2) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    mask = mask.float()
    k = 2 * dilation + 1
    pad = dilation

    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)
    edge_gt = (dilated - eroded).clamp(0.0, 1.0)
    return edge_gt


class EdgeLoss(nn.Module):
    def __init__(self,
                 bce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 pos_weight: float = 10.0,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer('pos_weight', torch.tensor(pos_weight))

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logit.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=logit.shape[2:], mode='nearest')

        bce = F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=self.pos_weight.to(logit.device))

        prob = torch.sigmoid(logit)
        intersection = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice

    def extra_repr(self) -> str:
        return (f'bce_weight={self.bce_weight}, dice_weight={self.dice_weight}, '
                f'pos_weight={self.pos_weight.item():.1f}')
