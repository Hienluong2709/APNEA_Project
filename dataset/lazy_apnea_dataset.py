import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LazyApneaSequenceDataset(Dataset):
    def __init__(self, block_dir, seq_len=10, return_name=False):
        self.block_dir = block_dir
        self.seq_len = seq_len
        self.return_name = return_name

        self.X_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("X_")])
        self.y_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("y_")])

        self.index_map = []
        for i, path in enumerate(self.X_paths):
            x = np.load(path, mmap_mode='r')
            for j in range(0, len(x) - seq_len + 1):
                self.index_map.append((i, j))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        block_idx, start_idx = self.index_map[idx]
        x_block = np.load(self.X_paths[block_idx], mmap_mode='r')
        y_block = np.load(self.y_paths[block_idx], mmap_mode='r')

        x_seq = x_block[start_idx:start_idx + self.seq_len]  # (T, 64, 684)
        y_seq = y_block[start_idx:start_idx + self.seq_len]  # (T,)

        x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(1)  # (T, 1, 64, 684)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)  # (T,)
        
        if self.return_name:
            block_name = os.path.basename(self.X_paths[block_idx])
            return x_tensor, y_tensor, block_name
        else:
            return x_tensor, y_tensor
