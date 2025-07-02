import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ApneaMelDataset(Dataset):
    def __init__(self, block_dir="data/blocks"):
        self.X_paths = sorted(glob.glob(os.path.join(block_dir, "X_*.npy")))
        self.y_paths = sorted(glob.glob(os.path.join(block_dir, "y_*.npy")))

        self.X = np.concatenate([np.load(p) for p in self.X_paths])
        self.y = np.concatenate([np.load(p) for p in self.y_paths])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, 64, 684)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
