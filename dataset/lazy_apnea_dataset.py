import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from scipy import signal

class AudioAugmentation:
    """
    Lớp augmentation cho dữ liệu âm thanh (mel-spectrogram)
    """
    def __init__(self, 
                 time_mask_prob=0.5, 
                 freq_mask_prob=0.5, 
                 time_mask_width=20, 
                 freq_mask_width=10,
                 noise_prob=0.3,
                 noise_level=0.05):
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        self.noise_prob = noise_prob
        self.noise_level = noise_level
    
    def add_time_mask(self, spec):
        """Thêm time mask vào mel-spectrogram"""
        time_length = spec.shape[1]
        mask_start = random.randint(0, time_length - self.time_mask_width)
        mask_end = min(mask_start + self.time_mask_width, time_length)
        spec_copy = spec.copy()
        spec_copy[:, mask_start:mask_end] = 0
        return spec_copy
    
    def add_freq_mask(self, spec):
        """Thêm frequency mask vào mel-spectrogram"""
        freq_length = spec.shape[0]
        mask_start = random.randint(0, freq_length - self.freq_mask_width)
        mask_end = min(mask_start + self.freq_mask_width, freq_length)
        spec_copy = spec.copy()
        spec_copy[mask_start:mask_end, :] = 0
        return spec_copy
    
    def add_gaussian_noise(self, spec):
        """Thêm gaussian noise vào mel-spectrogram"""
        noise = np.random.normal(0, self.noise_level, spec.shape)
        return spec + noise
    
    def __call__(self, spec):
        """Áp dụng augmentation với xác suất cho trước"""
        if random.random() < self.time_mask_prob:
            spec = self.add_time_mask(spec)
        
        if random.random() < self.freq_mask_prob:
            spec = self.add_freq_mask(spec)
            
        if random.random() < self.noise_prob:
            spec = self.add_gaussian_noise(spec)
            
        return spec

class LazyApneaSequenceDataset(Dataset):
    def __init__(self, block_dir, seq_len=10, return_name=False, augment=False):
        self.block_dir = block_dir
        self.seq_len = seq_len
        self.return_name = return_name
        self.augment = augment
        self.augmentor = AudioAugmentation() if augment else None

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

        x_seq = x_block[start_idx:start_idx + self.seq_len].copy()  # (T, 64, 684)
        y_seq = y_block[start_idx:start_idx + self.seq_len].copy()  # (T,)

        # Áp dụng augmentation nếu cần
        if self.augment:
            for i in range(len(x_seq)):
                x_seq[i] = self.augmentor(x_seq[i])

        x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(1)  # (T, 1, 64, 684)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)  # (T,)
        
        if self.return_name:
            block_name = os.path.basename(self.X_paths[block_idx])
            return x_tensor, y_tensor, block_name
        else:
            return x_tensor, y_tensor

class LazyApneaSingleDataset(Dataset):
    def __init__(self, block_dir, return_name=False, augment=False, balance_classes=False, upsample_minority=1.0):
        self.block_dir = block_dir
        self.return_name = return_name
        self.augment = augment
        self.augmentor = AudioAugmentation() if augment else None
        self.balance_classes = balance_classes
        self.upsample_minority = upsample_minority

        self.X_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("X_")])
        self.y_paths = sorted([os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.startswith("y_")])

        self.index_map = []
        self.label_counts = {0: 0, 1: 0}  # Đếm số lượng mẫu mỗi lớp
        
        for i, path in enumerate(self.X_paths):
            x_file = np.load(path, mmap_mode='r')
            y_file = np.load(self.y_paths[i], mmap_mode='r')
            
            for j in range(len(x_file)):
                self.index_map.append((i, j))
                
                # Đếm nhãn
                label = y_file[j]
                if label < len(self.label_counts):
                    self.label_counts[label] += 1
        
        # Upsampling lớp thiểu số
        if balance_classes and upsample_minority > 0:
            self._balance_dataset()
            
        print(f"Phân bố nhãn trong dataset: {self.label_counts}")

    def _balance_dataset(self):
        """Cân bằng tập dữ liệu bằng cách nhân bản các mẫu của lớp thiểu số"""
        # Tìm lớp thiểu số
        minority_class = min(self.label_counts, key=self.label_counts.get)
        majority_class = 1 - minority_class
        
        # Tỷ lệ upsampling
        upsampling_ratio = int(self.label_counts[majority_class] / self.label_counts[minority_class] * self.upsample_minority)
        
        if upsampling_ratio <= 1:
            return
        
        # Tìm các mẫu của lớp thiểu số
        minority_samples = []
        for idx, (block_idx, sample_idx) in enumerate(self.index_map):
            try:
                y_block = np.load(self.y_paths[block_idx], mmap_mode='r')
                label = y_block[sample_idx]
                
                if label == minority_class:
                    minority_samples.append((block_idx, sample_idx))
            except Exception as e:
                print(f"Lỗi khi kiểm tra nhãn {block_idx}, {sample_idx}: {e}")
        
        # Thêm nhiều bản sao của các mẫu thiểu số
        additional_samples = []
        for _ in range(upsampling_ratio - 1):
            additional_samples.extend(minority_samples)
        
        self.index_map.extend(additional_samples)
        self.label_counts[minority_class] += len(additional_samples)
        
        # Xáo trộn lại index_map
        random.shuffle(self.index_map)
        
        print(f"Đã upsampling lớp {minority_class} với tỷ lệ {upsampling_ratio}x")
        print(f"Phân bố nhãn sau khi cân bằng: {self.label_counts}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        block_idx, sample_idx = self.index_map[idx]
        
        try:
            x_block = np.load(self.X_paths[block_idx], mmap_mode='r')
            y_block = np.load(self.y_paths[block_idx], mmap_mode='r')

            x_sample = x_block[sample_idx].copy()  # (64, 684)
            y_sample = y_block[sample_idx]  # scalar

            # Áp dụng augmentation nếu cần
            if self.augment:
                x_sample = self.augmentor(x_sample)

            # Thêm chiều channel (C, H, W)
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)  # (1, 64, 684)
            y_tensor = torch.tensor(y_sample, dtype=torch.long)  # scalar
            
            if self.return_name:
                block_name = os.path.basename(self.X_paths[block_idx])
                return x_tensor, y_tensor, block_name
            else:
                return x_tensor, y_tensor
        except Exception as e:
            print(f"Error loading sample {idx} from block {block_idx}, sample {sample_idx}: {e}")
            # Trả về mẫu hợp lệ
            return torch.zeros((1, 64, 684), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

class MixUpDataset(Dataset):
    """
    Dataset áp dụng kỹ thuật MixUp để tăng cường học tập
    """
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x1, y1 = self.dataset[idx]
        
        # Chọn ngẫu nhiên một mẫu thứ hai
        idx2 = random.randint(0, len(self.dataset) - 1)
        if idx2 == idx:
            idx2 = (idx2 + 1) % len(self.dataset)
            
        x2, y2 = self.dataset[idx2]
        
        # Tạo hệ số pha trộn
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Trộn dữ liệu và nhãn
        x = lam * x1 + (1 - lam) * x2
        
        return x, y1, y2, lam
