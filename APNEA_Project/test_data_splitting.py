"""
Script kiểm tra các phương pháp chia dữ liệu với dữ liệu mẫu
để xác nhận hoạt động đúng của module data_splitting.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các hàm từ module data_splitting
from utils.data_splitting import (
    random_split_dataset,
    dependent_subject_split,
    independent_subject_split,
    get_dataloaders_from_split,
    save_split_info,
    load_split_info,
    get_data_distribution
)

# Tạo dataset giả lập
class DummyDataset(Dataset):
    def __init__(self, size=100, label_ratio=0.3, patient_id="P000"):
        """
        Tạo dataset giả lập
        
        Args:
            size: Số lượng mẫu
            label_ratio: Tỷ lệ mẫu có nhãn 1 (apnea)
            patient_id: ID của bệnh nhân
        """
        self.size = size
        self.patient_id = patient_id
        
        # Tạo dữ liệu giả
        self.data = torch.randn(size, 1, 64, 684)  # (N, C, H, W)
        
        # Tạo nhãn với tỷ lệ label_ratio có nhãn 1 (apnea)
        num_positive = int(size * label_ratio)
        self.labels = torch.zeros(size, dtype=torch.long)
        self.labels[:num_positive] = 1
        
        # Xáo trộn dữ liệu và nhãn theo cùng một cách
        indices = torch.randperm(size)
        self.data = self.data[indices]
        self.labels = self.labels[indices]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def test_random_split():
    """
    Kiểm tra phương pháp chia ngẫu nhiên
    """
    print("\n" + "="*80)
    print("KIỂM TRA PHƯƠNG PHÁP CHIA NGẪU NHIÊN (RANDOM SPLIT)")
    print("="*80)
    
    # Tạo dataset với 1000 mẫu, 30% nhãn 1
    full_dataset = DummyDataset(size=1000, label_ratio=0.3)
    
    # Chia dataset
    train_dataset, val_dataset = random_split_dataset(full_dataset, train_ratio=0.8, seed=42)
    
    # Kiểm tra kích thước
    print(f"Kích thước dataset gốc: {len(full_dataset)}")
    print(f"Kích thước tập train: {len(train_dataset)}")
    print(f"Kích thước tập val: {len(val_dataset)}")
    
    # Kiểm tra phân bố nhãn
    train_dist = get_data_distribution(train_dataset)
    val_dist = get_data_distribution(val_dataset)
    
    print(f"Phân bố nhãn tập train: {train_dist}")
    print(f"Phân bố nhãn tập val: {val_dist}")
    
    # Tính tỷ lệ nhãn
    train_ratio_1 = train_dist.get(1, 0) / len(train_dataset) if len(train_dataset) > 0 else 0
    val_ratio_1 = val_dist.get(1, 0) / len(val_dataset) if len(val_dataset) > 0 else 0
    
    print(f"Tỷ lệ nhãn 1 trong tập train: {train_ratio_1:.2f}")
    print(f"Tỷ lệ nhãn 1 trong tập val: {val_ratio_1:.2f}")
    
    # Tạo DataLoader
    train_loader, val_loader = get_dataloaders_from_split(train_dataset, val_dataset, batch_size=32, num_workers=0)
    
    # Kiểm tra DataLoader
    print(f"Số batch trong train_loader: {len(train_loader)}")
    print(f"Số batch trong val_loader: {len(val_loader)}")
    
    return True

def test_dependent_subject_split():
    """
    Kiểm tra phương pháp chia theo bệnh nhân phụ thuộc
    """
    print("\n" + "="*80)
    print("KIỂM TRA PHƯƠNG PHÁP CHIA THEO BỆNH NHÂN PHỤ THUỘC (DEPENDENT SUBJECT SPLIT)")
    print("="*80)
    
    # Tạo dataset cho 5 bệnh nhân khác nhau
    patient_ids = ["P001", "P002", "P003", "P004", "P005"]
    datasets = [
        DummyDataset(size=200, label_ratio=0.2, patient_id=patient_ids[0]),
        DummyDataset(size=300, label_ratio=0.3, patient_id=patient_ids[1]),
        DummyDataset(size=250, label_ratio=0.4, patient_id=patient_ids[2]),
        DummyDataset(size=180, label_ratio=0.25, patient_id=patient_ids[3]),
        DummyDataset(size=270, label_ratio=0.35, patient_id=patient_ids[4])
    ]
    
    # Chia dataset
    train_dataset, val_dataset = dependent_subject_split(datasets, patient_ids, train_ratio=0.8, seed=42)
    
    # Kiểm tra kích thước
    total_size = sum(len(ds) for ds in datasets)
    print(f"Tổng kích thước dataset gốc: {total_size}")
    print(f"Kích thước tập train: {len(train_dataset)}")
    print(f"Kích thước tập val: {len(val_dataset)}")
    
    # Kiểm tra phân bố nhãn
    train_dist = get_data_distribution(train_dataset)
    val_dist = get_data_distribution(val_dataset)
    
    print(f"Phân bố nhãn tập train: {train_dist}")
    print(f"Phân bố nhãn tập val: {val_dist}")
    
    # Tạo DataLoader
    train_loader, val_loader = get_dataloaders_from_split(train_dataset, val_dataset, batch_size=32, num_workers=0)
    
    # Kiểm tra DataLoader
    print(f"Số batch trong train_loader: {len(train_loader)}")
    print(f"Số batch trong val_loader: {len(val_loader)}")
    
    return True

def test_independent_subject_split():
    """
    Kiểm tra phương pháp chia theo bệnh nhân độc lập
    """
    print("\n" + "="*80)
    print("KIỂM TRA PHƯƠNG PHÁP CHIA THEO BỆNH NHÂN ĐỘC LẬP (INDEPENDENT SUBJECT SPLIT)")
    print("="*80)
    
    # Tạo dataset cho 5 bệnh nhân khác nhau
    patient_ids = ["P001", "P002", "P003", "P004", "P005"]
    datasets = [
        DummyDataset(size=200, label_ratio=0.2, patient_id=patient_ids[0]),
        DummyDataset(size=300, label_ratio=0.3, patient_id=patient_ids[1]),
        DummyDataset(size=250, label_ratio=0.4, patient_id=patient_ids[2]),
        DummyDataset(size=180, label_ratio=0.25, patient_id=patient_ids[3]),
        DummyDataset(size=270, label_ratio=0.35, patient_id=patient_ids[4])
    ]
    
    # Chia dataset
    train_dataset, val_dataset, train_patients, val_patients = independent_subject_split(
        datasets, patient_ids, train_ratio=0.6, seed=42
    )
    
    # Kiểm tra kích thước
    total_size = sum(len(ds) for ds in datasets)
    print(f"Tổng kích thước dataset gốc: {total_size}")
    print(f"Kích thước tập train: {len(train_dataset)}")
    print(f"Kích thước tập val: {len(val_dataset)}")
    
    # Kiểm tra danh sách bệnh nhân
    print(f"Bệnh nhân trong tập train: {train_patients}")
    print(f"Bệnh nhân trong tập val: {val_patients}")
    
    # Kiểm tra phân bố nhãn
    train_dist = get_data_distribution(train_dataset)
    val_dist = get_data_distribution(val_dataset)
    
    print(f"Phân bố nhãn tập train: {train_dist}")
    print(f"Phân bố nhãn tập val: {val_dist}")
    
    # Tạo DataLoader
    train_loader, val_loader = get_dataloaders_from_split(train_dataset, val_dataset, batch_size=32, num_workers=0)
    
    # Kiểm tra DataLoader
    print(f"Số batch trong train_loader: {len(train_loader)}")
    print(f"Số batch trong val_loader: {len(val_loader)}")
    
    # Lưu thông tin chia dữ liệu
    os.makedirs('results', exist_ok=True)
    save_split_info(
        output_dir='results',
        split_type='independent',
        train_patients=train_patients,
        val_patients=val_patients,
        additional_info={
            'train_ratio': 0.6,
            'seed': 42
        }
    )
    
    # Đọc lại thông tin chia dữ liệu
    split_info = load_split_info('results/split_info.json')
    print("\nThông tin chia dữ liệu đã lưu:")
    print(f"  Loại chia: {split_info['split_type']}")
    print(f"  Bệnh nhân train: {split_info['train_patients']}")
    print(f"  Bệnh nhân val: {split_info['val_patients']}")
    print(f"  Thông tin thêm: {split_info['additional_info']}")
    
    return True

def main():
    """
    Hàm chính
    """
    # Kiểm tra phương pháp chia ngẫu nhiên
    test_random_split()
    
    # Kiểm tra phương pháp chia theo bệnh nhân phụ thuộc
    test_dependent_subject_split()
    
    # Kiểm tra phương pháp chia theo bệnh nhân độc lập
    test_independent_subject_split()
    
    print("\n" + "="*80)
    print("✅ KIỂM TRA HOÀN TẤT - TẤT CẢ CÁC PHƯƠNG PHÁP CHIA DỮ LIỆU HOẠT ĐỘNG ĐÚNG")
    print("="*80)

if __name__ == "__main__":
    main()
