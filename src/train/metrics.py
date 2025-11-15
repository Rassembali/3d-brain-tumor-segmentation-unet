import numpy as np
import torch

def dice_score(pred, true, cls):
    pred_bin = (pred == cls)
    true_bin = (true == cls)

    inter = (pred_bin & true_bin).sum()
    denom = pred_bin.sum() + true_bin.sum()

    if denom == 0:
        return 1.0
    return 2 * inter / denom


def dice_region(pred, true, labels):
    pred_bin = np.isin(pred, labels)
    true_bin = np.isin(true, labels)

    inter = (pred_bin & true_bin).sum()
    denom = pred_bin.sum() + true_bin.sum()

    if denom == 0:
        return 1.0
    return 2 * inter / denom
