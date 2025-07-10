import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LazyApneaDataset(Dataset):
    """
    Phiên bản đơn giản của LazyApneaDataset, trả về một frame duy nhất
    thay vì một chuỗi các frame.
    """
    def __init__(self, block_dir, return_name=False):
        self.block_dir = block_dir
        self.return_name = return_name

        self.X_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("X_")])
        self.y_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("y_")])

        self.index_map = []
        for i, path in enumerate(self.X_paths):
            try:
                x = np.load(path, mmap_mode='r')
                for j in range(len(x)):
                    self.index_map.append((i, j))
            except Exception as e:
                print(f"Lỗi khi tải {path}: {e}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        try:
            block_idx, frame_idx = self.index_map[idx]
            x_block = np.load(self.X_paths[block_idx], mmap_mode='r')
            y_block = np.load(self.y_paths[block_idx], mmap_mode='r')

            x_frame = x_block[frame_idx]  # (64, 684)
            y_frame = y_block[frame_idx]  # ()

            x_tensor = torch.tensor(x_frame, dtype=torch.float32).unsqueeze(0)  # (1, 64, 684)
            y_tensor = torch.tensor(y_frame, dtype=torch.long)  # ()
            
            if self.return_name:
                block_name = os.path.basename(self.X_paths[block_idx])
                return x_tensor, y_tensor, block_name
            else:
                return x_tensor, y_tensor
        except Exception as e:
            print(f"Lỗi khi lấy mẫu {idx}: {e}")
            # Trả về tensor trống nếu có lỗi
            return torch.zeros((1, 64, 684), dtype=torch.float32), torch.tensor(0, dtype=torch.long)
