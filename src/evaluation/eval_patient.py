import os
from src.inference.predict import load_patient, predict_volume
from src.train.metrics import dice_score, dice_region
from src.inference.visualization import (
    visualize_middle_slice,
    visualize_patient_slices,
    visualize_tumor_slices_separate
)

def evaluate_patient(model, processed_dir, patient_id, save_images=True):

    X, Y = load_patient(processed_dir, patient_id)
    pred_mask = predict_volume(model, X)

    # Dice per class
    for c in [0,1,2,3]:
        d = dice_score(pred_mask, Y, c)
        print(f"Dice class {c}: {d:.4f}")

    # Region-wise Dice
    print("WT Dice:", dice_region(pred_mask, Y, [1,2,3]))
    print("TC Dice:", dice_region(pred_mask, Y, [2,3]))
    print("ET Dice:", dice_region(pred_mask, Y, [3]))

    if save_images:
        visualize_middle_slice(X, Y, pred_mask, "samples/middle_slice.png")
        visualize_patient_slices(X, Y, pred_mask, save_path="samples/grid.png")
        visualize_tumor_slices_separate(X, Y, pred_mask, out_dir="samples/tumor")

    return pred_mask
