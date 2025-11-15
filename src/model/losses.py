import torch
import torch.nn as nn

def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)
    dice = 0
    for c in range(4):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += 1 - (2 * inter + eps) / (union + eps)
    return dice / 4


ce_loss = nn.CrossEntropyLoss()

def combined_loss(pred, target):
    return 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
