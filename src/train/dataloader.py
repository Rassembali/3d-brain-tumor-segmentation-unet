import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSDataset(Dataset):
    def __init__(self, ids, processed_dir):
        self.ids = ids
        self.dir = processed_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        X = np.load(os.path.join(self.dir, f"{sid}_X.npy"))
        Y = np.load(os.path.join(self.dir, f"{sid}_Y.npy"))
        return torch.tensor(X), torch.tensor(Y)


def get_dataloader(processed_dir, batch_size=1, num_workers=2, split_ratio=0.8):
    subject_ids = sorted([
        f.replace("_X.npy", "") 
        for f in os.listdir(processed_dir) 
        if f.endswith("_X.npy")
    ])

    split = int(split_ratio * len(subject_ids))
    train_ids = subject_ids[:split]
    val_ids = subject_ids[split:]

    train_set = BraTSDataset(train_ids, processed_dir)
    val_set   = BraTSDataset(val_ids, processed_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ids, val_ids
