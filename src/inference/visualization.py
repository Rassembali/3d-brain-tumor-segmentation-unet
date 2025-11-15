import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_middle_slice(X, Y, pred_mask, save_path=None):
    mid = 64
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(X[0, mid], cmap="gray")
    plt.title("FLAIR")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(Y[mid], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_mask[mid], cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def visualize_patient_slices(X, Y, pred_mask, num_slices=12, save_path=None):
    slices = np.linspace(10, 118, num_slices).astype(int)
    plt.figure(figsize=(15, num_slices * 2))

    for i, s in enumerate(slices):
        plt.subplot(num_slices, 3, 3*i + 1)
        plt.imshow(X[0,s], cmap="gray")
        plt.title(f"FLAIR {s}")
        plt.axis("off")

        plt.subplot(num_slices, 3, 3*i + 2)
        plt.imshow(Y[s], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_slices, 3, 3*i + 3)
        plt.imshow(pred_mask[s], cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def visualize_tumor_slices_separate(X, Y, pred_mask, out_dir="samples/tumor"):
    os.makedirs(out_dir, exist_ok=True)

    tumor_slices = np.where(Y.sum(axis=(1,2)) > 0)[0]

    if len(tumor_slices) == 0:
        print("No tumor in this patient.")
        return

    for s in tumor_slices:
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(X[0,s], cmap="gray")
        plt.title(f"FLAIR slice {s}")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(Y[s], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(pred_mask[s], cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(f"{out_dir}/slice_{s}.png", dpi=300)
        plt.close()

    print(f"Saved {len(tumor_slices)} tumor slices to {out_dir}")
