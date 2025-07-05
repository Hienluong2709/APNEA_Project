import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class PatientBlockDataset(Dataset):
    def __init__(self, block_dir, seq_len=5):
        self.seq_len = seq_len
        self.X_paths = []
        self.y_paths = []
        self.index_map = []  # (block_index, item_start_index)

        x_files = sorted(glob.glob(os.path.join(block_dir, "X_*.npy")))
        y_files = sorted(glob.glob(os.path.join(block_dir, "y_*.npy")))

        for x_path, y_path in zip(x_files, y_files):
            x_data = np.load(x_path, mmap_mode='r')
            y_data = np.load(y_path, mmap_mode='r')

            if len(x_data) < self.seq_len:
                continue  # Bỏ qua block quá ngắn

            self.X_paths.append(x_path)
            self.y_paths.append(y_path)

            block_index = len(self.X_paths) - 1
            print(f"✅ Đã load block từ {block_dir}, shape: {x_data.shape}")

            for i in range(len(x_data) - self.seq_len + 1):
                self.index_map.append((block_index, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        block_idx, start_idx = self.index_map[idx]
        x_data = np.load(self.X_paths[block_idx], mmap_mode='r')
        y_data = np.load(self.y_paths[block_idx], mmap_mode='r')

        x_seq = x_data[start_idx : start_idx + self.seq_len]  # [seq_len, H, W] hoặc [seq_len, C, H, W]
        y_label = y_data[start_idx + self.seq_len - 1]

        # Nếu mỗi ảnh chỉ có 2D (H, W) → thêm chiều channel C=1
        if len(x_seq.shape) == 3:  # [seq_len, H, W]
            x_seq = np.expand_dims(x_seq, axis=1)  # → [seq_len, 1, H, W]

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)  # [seq_len, C, H, W]
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        return x_tensor, y_tensor

