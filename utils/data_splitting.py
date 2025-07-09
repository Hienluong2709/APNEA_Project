"""
Các phương pháp chia dữ liệu cho mô hình phát hiện apnea:
- Dependent subject: Chia dữ liệu của từng bệnh nhân thành train/val
- Independent subject: Chia bệnh nhân thành nhóm train/val riêng biệt
- Random split: Chia ngẫu nhiên toàn bộ dữ liệu không phân biệt bệnh nhân
"""

import os
import random
import torch
from torch.utils.data import Subset, ConcatDataset, random_split, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any


def random_split_dataset(full_dataset, train_ratio=0.8, seed=42):
    """
    Chia dữ liệu ngẫu nhiên thành tập train và validation
    
    Args:
        full_dataset: Dataset gốc
        train_ratio: Tỷ lệ dữ liệu dùng cho huấn luyện (mặc định: 0.8)
        seed: Seed ngẫu nhiên cho khả năng tái tạo kết quả
        
    Returns:
        train_dataset, val_dataset: Hai dataset cho train và validation
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Split sizes
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Random split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Random Split: {len(train_dataset)} mẫu train, {len(val_dataset)} mẫu validation")
    
    return train_dataset, val_dataset


def dependent_subject_split(datasets, patient_ids, train_ratio=0.8, seed=42):
    """
    Chia dữ liệu theo phương pháp dependent subject: 
    Dữ liệu của mỗi bệnh nhân được chia thành tập train và val
    
    Args:
        datasets: List của các dataset, mỗi dataset tương ứng một bệnh nhân
        patient_ids: List ID của các bệnh nhân
        train_ratio: Tỷ lệ dữ liệu dùng cho huấn luyện (mặc định: 0.8)
        seed: Seed ngẫu nhiên cho khả năng tái tạo kết quả
        
    Returns:
        train_dataset, val_dataset: Hai dataset cho train và validation
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    train_subsets = []
    val_subsets = []
    
    for ds, patient_id in zip(datasets, patient_ids):
        # Chia dữ liệu của từng bệnh nhân
        indices = list(range(len(ds)))
        random.shuffle(indices)
        
        split_idx = int(train_ratio * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_subsets.append(Subset(ds, train_indices))
        val_subsets.append(Subset(ds, val_indices))
    
    # Kết hợp các subset
    train_dataset = ConcatDataset(train_subsets)
    val_dataset = ConcatDataset(val_subsets)
    
    print(f"Dependent Subject Split: {len(train_dataset)} mẫu train, {len(val_dataset)} mẫu validation")
    print(f"  (Mỗi bệnh nhân được chia thành {int(train_ratio*100)}% train và {int((1-train_ratio)*100)}% validation)")
    
    return train_dataset, val_dataset


def independent_subject_split(datasets, patient_ids, train_ratio=0.8, seed=42, stratify=None):
    """
    Chia dữ liệu theo phương pháp independent subject: 
    Các bệnh nhân được chia thành 2 nhóm riêng biệt cho tập train và val
    
    Args:
        datasets: List của các dataset, mỗi dataset tương ứng một bệnh nhân
        patient_ids: List ID của các bệnh nhân
        train_ratio: Tỷ lệ bệnh nhân dùng cho huấn luyện (mặc định: 0.8)
        seed: Seed ngẫu nhiên cho khả năng tái tạo kết quả
        stratify: Nếu cung cấp, dùng để phân tầng (ví dụ: mức độ apnea)
        
    Returns:
        train_dataset, val_dataset: Hai dataset cho train và validation
        train_patients, val_patients: Danh sách bệnh nhân trong mỗi tập
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    # Create list of patient indices
    patient_indices = list(range(len(patient_ids)))
    random.shuffle(patient_indices)
    
    # Split patients
    split_idx = int(train_ratio * len(patient_indices))
    train_patient_indices = patient_indices[:split_idx]
    val_patient_indices = patient_indices[split_idx:]
    
    # Get corresponding datasets
    train_datasets = [datasets[i] for i in train_patient_indices]
    val_datasets = [datasets[i] for i in val_patient_indices]
    
    # Get patient IDs
    train_patients = [patient_ids[i] for i in train_patient_indices]
    val_patients = [patient_ids[i] for i in val_patient_indices]
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    
    train_size = len(train_dataset) if train_dataset else 0
    val_size = len(val_dataset) if val_dataset else 0
    
    print(f"Independent Subject Split:")
    print(f"  Train: {len(train_patients)} bệnh nhân với {train_size} mẫu")
    print(f"  Validation: {len(val_patients)} bệnh nhân với {val_size} mẫu")
    print(f"  Bệnh nhân train: {', '.join(train_patients)}")
    print(f"  Bệnh nhân validation: {', '.join(val_patients)}")
    
    return train_dataset, val_dataset, train_patients, val_patients


def get_dataloaders_from_split(train_dataset, val_dataset, batch_size=32, num_workers=0, seed=42):
    """
    Tạo DataLoader từ các dataset đã chia
    
    Args:
        train_dataset: Dataset huấn luyện
        val_dataset: Dataset kiểm tra
        batch_size: Kích thước batch
        num_workers: Số luồng đọc dữ liệu
        seed: Seed ngẫu nhiên
        
    Returns:
        train_loader, val_loader: DataLoader cho train và validation
    """
    # Set seed for worker initialization
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=lambda id: np.random.seed(seed + id),
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def save_split_info(output_dir, split_type, train_patients=None, val_patients=None, additional_info=None):
    """
    Lưu thông tin về cách chia dữ liệu để tái sử dụng
    
    Args:
        output_dir: Thư mục để lưu
        split_type: Loại chia dữ liệu ('random', 'dependent', 'independent')
        train_patients: Danh sách ID bệnh nhân trong tập train (cho independent split)
        val_patients: Danh sách ID bệnh nhân trong tập val (cho independent split)
        additional_info: Thông tin thêm để lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    
    split_info = {
        'split_type': split_type,
        'train_patients': train_patients,
        'val_patients': val_patients,
        'additional_info': additional_info
    }
    
    import json
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    print(f"Đã lưu thông tin chia dữ liệu tại {os.path.join(output_dir, 'split_info.json')}")


def load_split_info(filepath):
    """
    Đọc thông tin về cách chia dữ liệu từ file
    
    Args:
        filepath: Đường dẫn tới file split_info.json
        
    Returns:
        split_info: Dict chứa thông tin về cách chia dữ liệu
    """
    import json
    with open(filepath, 'r') as f:
        split_info = json.load(f)
    
    return split_info


# Hàm tiện ích để lấy độ phân bố dữ liệu
def get_data_distribution(dataset):
    """
    Lấy phân bố của các nhãn trong dataset
    
    Args:
        dataset: Dataset cần kiểm tra
        
    Returns:
        distribution: Dict chứa số lượng mẫu cho mỗi nhãn
    """
    labels = []
    
    # Lấy nhãn từ dataset
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    # Đếm số lượng mẫu cho mỗi nhãn
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    distribution = {}
    for label, count in zip(unique_labels, counts):
        distribution[int(label)] = int(count)
    
    return distribution
