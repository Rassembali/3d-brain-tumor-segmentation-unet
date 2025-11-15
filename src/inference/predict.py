import numpy as np
import torch
from torch.amp import autocast

def load_patient(processed_dir, patient_id):
    X = np.load(f"{processed_dir}/{patient_id}_X.npy")
    Y = np.load(f"{processed_dir}/{patient_id}_Y.npy")
    return X, Y

def predict_volume(model, X, device="cuda"):
    model.eval()

    X_tensor = torch.tensor(X).unsqueeze(0).float().to(device)

    with torch.no_grad(), autocast("cuda"):
        logits = model(X_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    return pred_mask
