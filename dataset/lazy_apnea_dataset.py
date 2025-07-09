import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class LazyApneaDataset(Dataset):
    def __init__(self, block_dir):
        self.X_paths = sorted(glob.glob(os.path.join(block_dir, "X_*.npy")))
        self.y_paths = sorted(glob.glob(os.path.join(block_dir, "y_*.npy")))
        
        if not self.X_paths:
            raise ValueError(f"❌ Không tìm thấy dữ liệu X_*.npy trong thư mục {block_dir}")
        if not self.y_paths:
            raise ValueError(f"❌ Không tìm thấy dữ liệu y_*.npy trong thư mục {block_dir}")
        
        print(f"📊 Tìm thấy {len(self.X_paths)} blocks dữ liệu")

        self.index_map = []  # lưu (index block, index trong block)

        for i, path in enumerate(self.X_paths):
            x = np.load(path, mmap_mode='r')
            print(f"✅ Đã load block {i} với shape: {x.shape}")
            for j in range(len(x)):
                self.index_map.append((i, j))
                
        print(f"📈 Tổng số mẫu: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        block_idx, item_idx = self.index_map[idx]
        x = np.load(self.X_paths[block_idx], mmap_mode='r')[item_idx]
        y = np.load(self.y_paths[block_idx], mmap_mode='r')[item_idx]
        
        # x có shape (64, 684), cần thêm chiều kênh để thành (1, 64, 684)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor

