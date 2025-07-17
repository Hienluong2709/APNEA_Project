"""
Script huấn luyện mô hình ConvNeXtTransformerLite và tính toán chỉ số AHI
theo phương pháp Dependent Subject với các kỹ thuật tiên tiến
"""
import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import traceback
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Thêm đường dẫn đến thư mục gốc
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Import từ utils
try:
    from utils.data_splitting import dependent_subject_split
    from utils.metrics import calculate_ahi_from_predictions, classify_osa_severity
    from utils.visualization import plot_confusion_matrix
except ImportError:
    print("⚠️ Không tìm thấy utils modules, sẽ tạo các hàm thay thế...")

# Import các module
from models.convnext_transformer_lite import ConvNeXtTransformerLite
try:
    from dataset.lazy_apnea_dataset import LazyApneaSingleDataset as LazyApneaDataset
    from dataset.lazy_apnea_dataset import MixUpDataset
except ImportError:
    from dataset.lazy_apnea_dataset import LazyApneaSequenceDataset as LazyApneaDataset


def evaluate(model, dataloader, device, name=""):
    """Đánh giá mô hình binary classification - format giống LSTM"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).float()
            
            # Ensure y is the right shape for binary classification
            if y.dim() > 1:
                y = y.squeeze()
            
            out = model(x).squeeze()
            
            # Convert to binary predictions (0 or 1)
            binary_preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(binary_preds)
            all_labels.extend(y.cpu().numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    print(f"📊 {name} - Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return acc, f1


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_simple(model, train_loader, val_loader, test_loader, device, epochs=30, lr=5e-5, resume_path=None):
    """Huấn luyện mô hình binary classification - format giống LSTM"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.008)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss

    best_f1 = 0
    start_epoch = 0
    patience = 10
    patience_counter = 0

    # Resume từ checkpoint nếu có
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resume from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_f1 = checkpoint.get('best_f1', 0)
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint with F1: {best_f1:.4f}")
        except Exception as e:
            print(f"⚠️ Cannot load checkpoint: {e}")

    for epoch in range(start_epoch, epochs):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y = x.to(device), y.to(device).float()  # Convert to float for BCE loss
            
            # Ensure y is the right shape for binary classification
            if y.dim() > 1:
                y = y.squeeze()
            
            optimizer.zero_grad()

            out = model(x).squeeze()  # Remove extra dimensions for binary output
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Convert to binary predictions (0 or 1)
            binary_preds = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            all_preds.extend(binary_preds)
            all_labels.extend(y.cpu().numpy().astype(int))

        train_f1 = f1_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, val_f1 = evaluate(model, val_loader, device, name="Validation")

        print(f"📊 Epoch {epoch+1}: Train F1={train_f1:.4f}, Train Acc={train_acc:.4f}")
        print(f"          Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")

        # Lưu checkpoint tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'train_f1': train_f1
            }
            torch.save(checkpoint, "checkpoints/ConvNeXtTransformerLite_best_f1.pth")
            print(f"💾 Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        # Lưu định kỳ mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_acc': val_acc
            }
            torch.save(checkpoint, f"checkpoints/transformer_epoch_{epoch+1}.pth")

        print(f"[Epoch {epoch + 1}] Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement")
            break

    print(f"\n✅ Huấn luyện hoàn tất. Best Val F1: {best_f1:.4f}")

    # Đánh giá trên tập test với model tốt nhất
    best_checkpoint = torch.load("checkpoints/ConvNeXtTransformerLite_best_f1.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_acc, test_f1 = evaluate(model, test_loader, device, name="Testing")
    
    print(f"\n📈 Final Results:")
    print(f"   - Best Validation Acc: {best_checkpoint.get('val_acc', 0):.4f}")
    print(f"   - Best Validation F1: {best_f1:.4f}")
    print(f"   - Test Acc: {test_acc:.4f}")
    print(f"   - Test F1: {test_f1:.4f}")

    return model, best_f1


def load_data(data_root, seq_len=5, batch_size=48):
    """Load dữ liệu - format giống LSTM"""
    patients = sorted(os.listdir(data_root))
    datasets = []

    print(f"📂 Đang load dữ liệu từ {data_root}...")
    for p in patients:
        p_dir = os.path.join(data_root, p)
        if os.path.isdir(p_dir):
            try:
                print(f"📥 Loading block: {p}")
                ds = LazyApneaDataset(p_dir)
                print(f"✅ Loaded {p} - Tổng sequence: {len(ds)}")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Lỗi với {p}: {e}")

    if not datasets:
        raise RuntimeError("❌ Không có block nào được load!")

    full_dataset = ConcatDataset(datasets)
    total_len = len(full_dataset)
    print(f"📊 Tổng số sequence: {total_len}")

    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


def predict_and_save_csv_per_block(model, data_root, device, seq_len=5):
    """Tạo CSV predictions - format giống LSTM"""
    model.eval()
    os.makedirs("predictions", exist_ok=True)
    patients = sorted(os.listdir(data_root))

    print(f"\n🧪 Lưu dự đoán nhị phân dưới dạng CSV cho từng block...")
    for p in patients:
        p_dir = os.path.join(data_root, p)
        if not os.path.isdir(p_dir):
            continue

        try:
            ds = LazyApneaDataset(p_dir)
            loader = DataLoader(ds, batch_size=48, shuffle=False)

            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    out = model(x)
                    preds = torch.argmax(out, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

            df = pd.DataFrame({
                "label_true": all_labels,
                "label_pred": all_preds,
            })
            df["correct"] = df["label_true"] == df["label_pred"]
            accuracy = df["correct"].mean()
            
            save_path = f"predictions/{p}_preds.csv"
            df.to_csv(save_path, index=False)
            print(f"✅ Đã lưu: {save_path} (Accuracy: {accuracy:.4f})")

        except Exception as e:
            print(f"⚠️ Lỗi với block {p}: {e}")


def main_simple():
    """Hàm main đơn giản giống LSTM"""
    print("🚀 ConvNeXt+Transformer Training với Binary Predictions (Dependent)")
    print("📋 Workflow: Train → Binary (0,1) → 2 CSV files → MAE/RMSE/PCC")
    
    # Lấy đường dẫn project_dir từ biến global đã định nghĩa ở đầu file
    data_path = os.path.join(project_dir, "data", "blocks")

    if not os.path.exists(data_path):
        raise RuntimeError(f"❌ Không tìm thấy thư mục: {data_path}")

    os.makedirs("checkpoints", exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")

    # Load data với Dependent Subject approach (shuffle tất cả patients)
    train_loader, val_loader, test_loader = load_data(data_path, seq_len=5, batch_size=32)
    
    # Khởi tạo model với parameters tối ưu - QUAN TRỌNG: num_classes=1 cho binary
    model = ConvNeXtTransformerLite(
        num_classes=1,     # 1 output cho binary classification
        embed_dim=160,
        num_heads=5,
        num_transformer_layers=4,
        dropout=0.08
    )
    
    # Model summary
    total_params = count_parameters(model)
    print(f"🏗️ Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print(f"  Architecture: ConvNeXt + Transformer → Binary output (0,1)")

    # STEP 1: Training
    print("\n🚀 STEP 1: Training model...")
    resume_ckpt = "checkpoints/ConvNeXtTransformerLite_best_f1.pth"
    model, best_f1 = train_simple(model, train_loader, val_loader, test_loader, device, epochs=30, lr=5e-5, resume_path=resume_ckpt)

    # Tạo full dataset cho predictions
    from dataset.lazy_apnea_dataset import LazyApneaDataset
    full_dataset = LazyApneaDataset(data_path, seq_len=5, use_cache=True)
    
    # STEP 2: Tạo 2 file CSV từ binary predictions
    print("\n📊 STEP 2: Tạo 2 CSV files từ binary predictions...")
    
    # File 1: Model binary predictions → AHI
    model_df, model_csv = create_model_predictions_csv(model, full_dataset, device)
    
    # File 2: True labels → True AHI
    psg_df, psg_csv = create_ahi_psg_csv(full_dataset)
    
    # STEP 3: So sánh 2 files và tính MAE, RMSE, PCC
    comparison_df, final_metrics = compare_files_and_calculate_metrics(model_csv, psg_csv)
    
    print("\n✅ HOÀN THÀNH! Binary predictions workflow (Dependent)!")
    print("🎯 Train → Binary (0,1) → CSV files → MAE/RMSE/PCC calculation")
    if final_metrics:
        print(f"🏆 Final PCC: {final_metrics['pcc']:.4f}")
    
    optimize_memory()
    """Giải phóng bộ nhớ cache và thu gom rác"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Thêm giải phóng bộ nhớ CUDA không sử dụng
    torch.cuda.empty_cache()
    
    # Đặt biến môi trường để giới hạn cache PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


def mixup_data(x, y, alpha=0.3, device='cuda'):
    """Thực hiện MixUp trên batch dữ liệu"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Tính loss với MixUp"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_balanced_sampler(dataset, num_classes=2):
    """
    Tạo một sampler cân bằng các lớp trong tập dữ liệu huấn luyện.
    """
    # Lấy tất cả các nhãn từ dataset
    labels = []
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for ds in dataset.datasets:
            for _, y in ds:
                labels.append(y.item())
    else:
        for _, y in dataset:
            labels.append(y.item())
    
    # Đếm số lượng mẫu mỗi lớp
    labels = np.array(labels)
    class_counts = [np.sum(labels == i) for i in range(num_classes)]
    num_samples = len(labels)
    
    # Tính trọng số cho từng mẫu
    weights = [0] * num_samples
    for idx, label in enumerate(labels):
        weights[idx] = 1.0 / class_counts[label]
    
    # Tạo WeightedRandomSampler
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )
    
    return sampler


def set_seed(seed=42):
    """Đặt seed cho tái tạo kết quả"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_ahi_from_predictions(true_labels, pred_labels, epoch_duration_seconds=30):
    """Tính AHI từ dự đoán"""
    # Tính thời gian ngủ tổng cộng (giờ)
    total_time_hours = (len(true_labels) * epoch_duration_seconds) / 3600
    
    # Đếm số sự kiện ngưng thở
    true_apnea_events = np.sum(true_labels == 1)
    pred_apnea_events = np.sum(pred_labels == 1)
    
    # Tính AHI
    true_ahi = true_apnea_events / total_time_hours if total_time_hours > 0 else 0
    pred_ahi = pred_apnea_events / total_time_hours if total_time_hours > 0 else 0
    
    # Tính các metrics khác
    mae = mean_absolute_error([true_ahi], [pred_ahi])
    rmse = np.sqrt(mean_squared_error([true_ahi], [pred_ahi]))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'true_events': true_apnea_events,
        'pred_events': pred_apnea_events,
        'total_time_hours': total_time_hours
    }
    
    return true_ahi, pred_ahi, metrics


# Helper functions cho binary prediction workflow
def calculate_ahi_from_binary_predictions(binary_array, epoch_duration_seconds=30):
    """
    Tính AHI từ chuỗi binary predictions
    AHI = (số epoch có apnea * epochs_per_hour) / total_hours
    """
    if len(binary_array) == 0:
        return 0.0
    
    epochs_per_hour = 3600 / epoch_duration_seconds  # 120 epochs/hour với 30s/epoch
    total_hours = len(binary_array) / epochs_per_hour
    apnea_events = np.sum(binary_array == 1)
    
    if total_hours <= 0:
        return 0.0
    
    ahi = apnea_events / total_hours
    return float(ahi)

def classify_osa_severity(ahi):
    """Phân loại mức độ nghiêm trọng OSA"""
    if ahi < 5:
        return 'Normal'
    elif ahi < 15:
        return 'Mild'
    elif ahi < 30:
        return 'Moderate'
    else:
        return 'Severe'


def dependent_subject_split_optimized(datasets, patient_ids, train_ratio=0.8, seed=42):
    """
    Chia dữ liệu theo phương pháp dependent subject tối ưu với tỷ lệ 80/10/10
    Mỗi bệnh nhân được chia thành train (80%), validation (10%) và test (10%)
    
    Args:
        datasets: List các dataset của từng bệnh nhân
        patient_ids: List ID bệnh nhân
        train_ratio: Tỷ lệ dữ liệu train (0.8 = 80%)
        seed: Seed cho random
        
    Returns:
        train_datasets, val_datasets, test_datasets: List các dataset cho mỗi tập
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    print("🔀 Chia dữ liệu theo dependent subject tối ưu (80/10/10)...")
    
    total_train_samples = 0
    total_val_samples = 0
    total_test_samples = 0
    
    for dataset, patient_id in zip(datasets, patient_ids):
        try:
            total_samples = len(dataset)
            
            # Tạo indices và shuffle
            indices = list(range(total_samples))
            random.shuffle(indices)
            
            # Chia theo tỷ lệ 80/10/10
            train_size = int(train_ratio * total_samples)
            remaining_size = total_samples - train_size
            
            # Chia phần remaining thành val và test (mỗi phần 10%)
            val_size = remaining_size // 2
            test_size = remaining_size - val_size
            
            # Tạo indices cho từng tập
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Tạo Subset
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            test_subset = torch.utils.data.Subset(dataset, test_indices)
            
            train_datasets.append(train_subset)
            val_datasets.append(val_subset)
            test_datasets.append(test_subset)
            
            # Cập nhật tổng số mẫu
            total_train_samples += train_size
            total_val_samples += val_size
            total_test_samples += test_size
            
            # In thông tin chi tiết
            train_pct = (train_size / total_samples) * 100
            val_pct = (val_size / total_samples) * 100
            test_pct = (test_size / total_samples) * 100
            
            print(f"  {patient_id}: {train_size}/{total_samples} ({train_pct:.1f}%) train, "
                  f"{val_size}/{total_samples} ({val_pct:.1f}%) val, "
                  f"{test_size}/{total_samples} ({test_pct:.1f}%) test")
            
            # Tối ưu bộ nhớ
            optimize_memory()
            
        except Exception as e:
            print(f"❌ Lỗi khi chia dữ liệu cho {patient_id}: {e}")
            continue
    
    print(f"\n📊 Tổng kết chia dữ liệu:")
    total_samples = total_train_samples + total_val_samples + total_test_samples
    print(f"  Train: {total_train_samples}/{total_samples} ({total_train_samples/total_samples*100:.1f}%)")
    print(f"  Validation: {total_val_samples}/{total_samples} ({total_val_samples/total_samples*100:.1f}%)")
    print(f"  Test: {total_test_samples}/{total_samples} ({total_test_samples/total_samples*100:.1f}%)")
    
    return train_datasets, val_datasets, test_datasets


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', 
                use_amp=True, use_mixup=False, use_swa=False, weight_decay=0.01,
                use_early_stopping=False, patience=5):
    """Huấn luyện mô hình đơn giản hóa"""
    print(f"🚀 Huấn luyện mô hình {model.__class__.__name__} trên {device}")
    
    # Tạo thư mục checkpoints
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{model.__class__.__name__}_best.pth')
    
    # Khởi tạo optimizer và criterion
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Sử dụng weighted loss nếu cần
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  Sử dụng weighted loss với weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler - OneCycleLR với cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, div_factor=10, final_div_factor=100, anneal_strategy='cos'
    )
    
    # GradScaler cho mixed precision
    scaler = GradScaler() if use_amp else None
    
    # Stochastic Weight Averaging
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_start = epochs // 3  # Bắt đầu SWA ở 1/3 quá trình
    
    best_val_f1 = 0
    no_improve_epochs = 0
    
    # Tối ưu bộ nhớ trước khi bắt đầu huấn luyện
    optimize_memory()
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        batch_count = 0
        
        for x, y in pbar:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)  # Tiết kiệm bộ nhớ hơn
            
            if use_mixup:
                # Áp dụng MixUp
                mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha, device=device)
                
                if use_amp:
                    with autocast():
                        outputs = model(mixed_x)
                        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    
                    # Backward and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(mixed_x)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                
                # Lấy dự đoán cho metrics (từ dữ liệu gốc)
                with torch.no_grad():
                    orig_outputs = model(x)
                    preds = orig_outputs.argmax(1).cpu().numpy()
            else:
                if use_amp:
                    with autocast():
                        outputs = model(x)
                        loss = criterion(outputs, y)
                    
                    # Backward and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                
                preds = outputs.argmax(1).cpu().numpy()
            
            # Update scheduler
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Xóa biến tạm để giải phóng bộ nhớ
            del x, y, outputs, loss, preds
            
            # Tối ưu bộ nhớ định kỳ để tránh tràn bộ nhớ
            batch_count += 1
            if batch_count % 20 == 0:  # Cứ mỗi 20 batch
                optimize_memory()
        
        # Calculate train metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Giải phóng bộ nhớ sau khi tính toán xong metrics
        del all_preds, all_labels
        optimize_memory()
        
        # Update SWA if enabled
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
        
        # VALIDATION
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
                
                # Xóa biến tạm để giải phóng bộ nhớ
                del x, y, outputs, loss, preds
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Giải phóng bộ nhớ sau khi tính toán xong metrics
        del all_preds, all_labels
        optimize_memory()
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Lưu mô hình tốt nhất với F1={val_f1:.4f}")
        
        # Print epoch results
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # Early stopping
        if use_early_stopping:
            if val_f1 > best_val_f1:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"  Không cải thiện F1 ({no_improve_epochs}/{patience} epochs)")
                
            if no_improve_epochs >= patience:
                print(f"⚠️ Early stopping tại epoch {epoch+1}/{epochs}")
                break
        
        # Tối ưu bộ nhớ ở cuối mỗi epoch
        optimize_memory()
    
    # Tối ưu bộ nhớ trước khi kết thúc
    optimize_memory()
    return model, best_val_f1


def create_model_predictions_csv(model, full_dataset, device):
    """
    BƯỚC 2A: Tạo File 1 - Model binary predictions → AHI cho từng patient
    """
    print("🔍 BƯỚC 2A: Tạo File 1 - Model predictions...")
    model.eval()
    
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    predictions = []
    
    with torch.no_grad():
        for patient_id in tqdm(full_dataset.patient_ids[:25], desc="Processing patients"):
            try:
                patient_data = full_dataset.get_patient_data(patient_id, limit_blocks=None)
                
                if not patient_data:
                    continue
                
                # Process all blocks for this patient
                patient_preds = []
                for block_data in patient_data:
                    try:
                        if len(block_data) < 3:
                            continue
                            
                        features = block_data[0]
                        if features is None or features.size == 0:
                            continue
                        
                        # Convert to tensor và predict
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                        
                        # Model prediction
                        with autocast():
                            outputs = model(features_tensor)
                            
                        # Convert to binary prediction (0 or 1)
                        pred_prob = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                        binary_pred = 1 if pred_prob > 0.5 else 0
                        patient_preds.append(binary_pred)
                        
                    except Exception as e:
                        continue
                
                # Calculate AHI from binary predictions
                if len(patient_preds) > 0:
                    binary_array = np.array(patient_preds)
                    predicted_ahi = calculate_ahi_from_binary_predictions(binary_array)
                    predicted_severity = classify_osa_severity(predicted_ahi)
                    
                    predictions.append({
                        'patient_id': patient_id,
                        'predicted_ahi': predicted_ahi,
                        'predicted_severity': predicted_severity,
                        'total_epochs': len(patient_preds),
                        'apnea_events_pred': np.sum(binary_array == 1),
                        'normal_events_pred': np.sum(binary_array == 0),
                        'total_sleep_hours': len(patient_preds) * 30 / 3600
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing {patient_id}: {e}")
    
    # Save FILE 1
    predictions_df = pd.DataFrame(predictions)
    predictions_csv = os.path.join(results_dir, 'model_predictions_dependent.csv')
    predictions_df.to_csv(predictions_csv, index=False)
    
    print(f"✅ FILE 1 saved: {predictions_csv}")
    print(f"  Format: Binary predictions → Predicted AHI cho {len(predictions)} patients")
    return predictions_df, predictions_csv

def create_ahi_psg_csv(full_dataset):
    """
    BƯỚC 2B: Tạo File 2 - True labels → True AHI cho từng patient  
    """
    print("🔍 BƯỚC 2B: Tạo File 2 - True AHI PSG...")
    
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    psg_results = []
    
    for patient_id in tqdm(full_dataset.patient_ids[:25], desc="Processing PSG"):
        try:
            patient_data = full_dataset.get_patient_data(patient_id, limit_blocks=None)
            
            if not patient_data:
                continue
            
            # Collect true labels cho patient này
            true_labels = []
            for block_data in patient_data:
                try:
                    if len(block_data) < 3:
                        continue
                        
                    label = block_data[1]  # True label
                    if label is not None:
                        true_labels.append(int(label))
                        
                except Exception as e:
                    continue
            
            if len(true_labels) > 0:
                # Convert true labels → true AHI
                true_array = np.array(true_labels)
                true_ahi = calculate_ahi_from_binary_predictions(true_array)
                true_severity = classify_osa_severity(true_ahi)
                
                psg_results.append({
                    'patient_id': patient_id,
                    'ahi_psg': true_ahi,
                    'severity_psg': true_severity,
                    'total_epochs': len(true_labels),
                    'apnea_events_true': np.sum(true_array == 1),
                    'normal_events_true': np.sum(true_array == 0),
                    'total_sleep_hours': len(true_labels) * 30 / 3600
                })
                
        except Exception as e:
            print(f"⚠️ Error processing {patient_id}: {e}")
    
    # Save FILE 2
    psg_df = pd.DataFrame(psg_results)
    psg_csv = os.path.join(results_dir, 'ahi_psg_dependent.csv')
    psg_df.to_csv(psg_csv, index=False)
    
    print(f"✅ FILE 2 saved: {psg_csv}")
    print(f"  Format: True labels → True AHI cho {len(psg_results)} patients")
    return psg_df, psg_csv

def compare_files_and_calculate_metrics(model_csv, psg_csv):
    """
    BƯỚC 3: Đọc 2 file CSV và tính MAE, RMSE, PCC
    """
    print("🔍 BƯỚC 3: So sánh 2 files và tính MAE, RMSE, PCC...")
    
    # Đọc 2 files
    model_df = pd.read_csv(model_csv)
    psg_df = pd.read_csv(psg_csv)
    
    # Merge theo patient_id
    merged_df = pd.merge(model_df, psg_df, on='patient_id', how='inner')
    
    if len(merged_df) == 0:
        print("❌ Không có patient nào match")
        return None, None
    
    print(f"✅ Matched {len(merged_df)} patients")
    
    # Extract AHI values
    predicted_ahi = merged_df['predicted_ahi'].values
    true_ahi = merged_df['ahi_psg'].values
    
    # Calculate metrics
    mae = mean_absolute_error(true_ahi, predicted_ahi)
    rmse = np.sqrt(mean_squared_error(true_ahi, predicted_ahi))
    
    # PCC (Pearson Correlation Coefficient)
    correlation_matrix = np.corrcoef(predicted_ahi, true_ahi)
    pcc = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    # Create comparison DataFrame
    comparison_df = merged_df[['patient_id', 'predicted_ahi', 'ahi_psg']].copy()
    comparison_df['absolute_error'] = np.abs(predicted_ahi - true_ahi)
    comparison_df['squared_error'] = (predicted_ahi - true_ahi) ** 2
    
    # Save comparison file
    results_dir = os.path.join(project_dir, 'results')
    comparison_csv = os.path.join(results_dir, 'ahi_comparison_dependent.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    
    # Final metrics
    final_metrics = {
        'mae': mae,
        'rmse': rmse,
        'pcc': pcc,
        'num_patients': len(merged_df)
    }
    
    print(f"📊 FINAL METRICS (Dependent Subject):")
    print(f"  👥 Patients: {final_metrics['num_patients']}")
    print(f"  📉 MAE: {final_metrics['mae']:.4f}")
    print(f"  📉 RMSE: {final_metrics['rmse']:.4f}")
    print(f"  🎯 PCC: {final_metrics['pcc']:.4f}")
    
    return comparison_df, final_metrics


def main():
    """Hàm chính thực hiện quá trình huấn luyện và đánh giá mô hình"""
    # Xác định đường dẫn dữ liệu
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    if args.data_dir:
        if os.path.isabs(args.data_dir):
            data_dir = args.data_dir
        else:
            data_dir = os.path.abspath(os.path.join(os.getcwd(), args.data_dir))
            if not os.path.exists(data_dir):
                data_dir = os.path.abspath(os.path.join(project_dir, args.data_dir))
    else:
        data_dir = os.path.join(project_dir, "data", "blocks")
    
    print(f"🔍 Đường dẫn dữ liệu: {data_dir}")
    
    # Tối ưu hóa bộ nhớ trước khi tải dữ liệu
    optimize_memory()
    
    # Kiểm tra tồn tại của thư mục dữ liệu
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục blocks tại {data_dir}")
        print(f"⚠️ Thư mục hiện tại: {os.getcwd()}")
        print("⚠️ Vui lòng chạy build_dataset.py trước hoặc kiểm tra lại đường dẫn")
        return
    
    # Tải dữ liệu
    try:
        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        print(f"📊 Đang tải dữ liệu từ {len(patient_dirs)} bệnh nhân")
        
        datasets = []
        patient_ids = []
        
        # Sắp xếp theo tên để đảm bảo tính nhất quán
        patient_dirs.sort()
        
        for p_dir in patient_dirs:
            try:
                patient_id = os.path.basename(p_dir)
                print(f"⏳ Đang tải dữ liệu từ {patient_id}...")
                
                # Tối ưu hóa bộ nhớ trước khi tải mỗi bệnh nhân
                optimize_memory()
                
                ds = LazyApneaDataset(p_dir)
                if len(ds) > 0:
                    datasets.append(ds)
                    patient_ids.append(patient_id)
                    print(f"✅ Đã tải {len(ds)} mẫu từ {patient_id}")
                else:
                    print(f"⚠️ Không có mẫu nào từ {patient_id}")
                
                # Kiểm tra bộ nhớ sau khi tải dữ liệu từ mỗi bệnh nhân
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    print(f"  Bộ nhớ CUDA sử dụng: {allocated:.2f} GB")
                
            except Exception as e:
                print(f"❌ Lỗi khi tải dữ liệu từ {p_dir}: {e}")
                print(f"  Chi tiết: {traceback.format_exc()}")
                
        print(f"✅ Đã tải dữ liệu từ {len(datasets)}/{len(patient_dirs)} bệnh nhân")
        
        # Tối ưu hóa bộ nhớ sau khi tải dữ liệu
        optimize_memory()
        
        if not datasets:
            print("❌ Không có dữ liệu để huấn luyện.")
            return
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        return
    
    # Chia dữ liệu theo phương pháp dependent subject tối ưu
    print("\n🔀 Đang chia dữ liệu theo phương pháp dependent subject tối ưu...")
    try:
        # Tối ưu hóa bộ nhớ trước khi chia dữ liệu
        optimize_memory()
        
        # Sử dụng hàm chia dữ liệu tối ưu với tỷ lệ 80/10/10
        train_datasets, val_datasets, test_datasets = dependent_subject_split_optimized(
            datasets, patient_ids, train_ratio=0.8, seed=42
        )
        
        # Tạo combined datasets
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)
        
        print(f"\n📈 Tổng số mẫu sau khi gộp: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
        
    except Exception as e:
        print(f"❌ Lỗi khi chia dữ liệu: {e}")
        return
    
    # Tạo DataLoader
    try:
        print("\n🔄 Đang tạo DataLoader...")
        
        # Tối ưu bộ nhớ trước khi tạo DataLoader
        optimize_memory()
        
        # Sử dụng pin_memory=False nếu không đủ RAM
        pin_memory = torch.cuda.is_available() and args.pin_memory
        persistent_workers = False if args.num_workers > 0 else False
        prefetch_factor = 2 if args.num_workers > 0 else None
        
        # Sử dụng sampler cân bằng nếu cần
        if args.balance_classes:
            train_sampler = get_balanced_sampler(train_dataset, num_classes=2)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
                drop_last=False
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
                drop_last=False
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
            drop_last=False
        )
        
        print(f"✅ Đã tạo DataLoader với batch_size={args.batch_size}, num_workers={args.num_workers}, pin_memory={pin_memory}")
        optimize_memory()  # Tối ưu bộ nhớ sau khi tạo xong DataLoaders
    except Exception as e:
        print(f"❌ Lỗi khi tạo DataLoader: {e}")
        print("  Thử giảm batch_size hoặc num_workers để tiết kiệm bộ nhớ")
        return
    
    # Chọn thiết bị
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n🖥️ Sử dụng thiết bị: {device}")
    
    # Tối ưu hóa bộ nhớ trước khi khởi tạo mô hình
    optimize_memory()
    
    # Khởi tạo mô hình với kích thước tối ưu để đạt 2.8M tham số
    try:
        # Cấu hình để có đúng khoảng 2.8M tham số - Tối ưu cho PCC cao
        model = ConvNeXtTransformerLite(
            num_classes=2, 
            embed_dim=160,                    # Tối ưu cho PCC: giảm xuống 160
            num_heads=5,                      # Tối ưu cho PCC: giảm xuống 5
            num_transformer_layers=4,         # Tối ưu cho PCC: giảm xuống 4 layers
            dropout=args.dropout,             # Dropout chính
            dropout_path=0.05                 # Giảm dropout_path để cải thiện PCC
        ).to(device)
        total_params = count_parameters(model)
        print(f"✅ Khởi tạo mô hình {model.__class__.__name__} thành công")
        print(f"📊 Mô hình có tổng cộng {total_params:,} tham số ({total_params/1e6:.2f}M)")
        print(f"  Embed dim: 160, Num heads: 5, Transformer layers: 4 (Tối ưu cho PCC)")
        print(f"  Dropout: {args.dropout}, Dropout path: 0.05, Weight decay: {args.weight_decay}")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo mô hình: {e}")
        return
    
    # Huấn luyện hoặc tải mô hình
    try:
        if not args.eval_only:
            print(f"\n🚀 Bắt đầu huấn luyện mô hình với {args.epochs} epochs...")
            model, best_val_f1 = train_model(
                model, train_loader, val_loader,
                epochs=args.epochs,
                lr=args.learning_rate,
                device=device,
                use_amp=args.use_amp,
                use_mixup=args.use_mixup,
                use_swa=args.use_swa,
                weight_decay=args.weight_decay,
                use_early_stopping=args.use_early_stopping,
                patience=args.patience
            )
            print(f"\n✅ Huấn luyện hoàn tất. F1 tốt nhất: {best_val_f1:.4f}")
        else:
            # Tải mô hình đã huấn luyện
            checkpoint_path = os.path.join(project_dir, 'checkpoints', f'{model.__class__.__name__}_best.pth')
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f"✅ Đã tải mô hình từ {checkpoint_path}")
            else:
                print(f"❌ Không tìm thấy checkpoint tại {checkpoint_path}")
                return
    except Exception as e:
        print(f"❌ Lỗi trong quá trình huấn luyện/tải mô hình: {e}")
        return
    
    # Tạo 2 file CSV riêng biệt và so sánh để cải thiện PCC
    try:
        print("\n📊 Tạo file CSV dự đoán mô hình và AHI PSG...")
        
        # Đảm bảo thư mục results tồn tại
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Tạo file CSV dự đoán mô hình
        model_csv_path = os.path.join(results_dir, 'model_predictions.csv')
        model_df = create_model_predictions_csv(model, datasets, patient_ids, device, model_csv_path)
        
        # 2. Tạo file CSV AHI PSG (ground truth)
        psg_csv_path = os.path.join(results_dir, 'ahi_psg.csv')
        psg_df = create_ahi_psg_csv(datasets, patient_ids, psg_csv_path)
        
        # 3. So sánh 2 file CSV để tính MAE, RMSE, PCC
        comparison_csv_path = os.path.join(results_dir, 'comparison_results.csv')
        comparison_df, metrics = compare_files_and_calculate_metrics(model_csv_path, psg_csv_path)
        
        if comparison_df is not None:
            print(f"\n🎯 SUMMARY METRICS (Dependent Subject):")
            print(f"  📈 MAE: {metrics['mae']:.2f}")
            print(f"  📈 RMSE: {metrics['rmse']:.2f}")  
            print(f"  📈 PCC: {metrics['pcc']:.4f}")
            print(f"  📈 R²: {metrics['r2']:.4f}")
            print(f"  📈 Severity Accuracy: {metrics['severity_acc']:.2f}")
            
            print(f"\n📁 FILES ĐƯỢC TẠO:")
            print(f"  📄 Dự đoán mô hình: {model_csv_path}")
            print(f"  📄 AHI PSG: {psg_csv_path}")
            print(f"  📄 So sánh chi tiết: {comparison_csv_path}")
        
        print("\n✅ Hoàn tất đánh giá mô hình Dependent Subject!")
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo CSV và so sánh: {e}")
        traceback.print_exc()


def optimize_memory():
    """Giải phóng bộ nhớ cache và thu gom rác"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Thêm giải phóng bộ nhớ CUDA không sử dụng
    torch.cuda.empty_cache()
    
    # Đặt biến môi trường để giới hạn cache PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


if __name__ == "__main__":
    try:
        # Chế độ đơn giản giống LSTM
        if len(sys.argv) == 1 or '--simple' in sys.argv:
            main_simple()
        else:
            # Chế độ nâng cao với argparse
            print("\n🚀 Bắt đầu quá trình huấn luyện mô hình ConvNeXtTransformerLite (Dependent Subject)...")
            parser = argparse.ArgumentParser(description='Huấn luyện mô hình ConvNeXtTransformerLite cho phát hiện ngưng thở khi ngủ')
            parser.add_argument('--data_dir', type=str, help='Đường dẫn đến thư mục dữ liệu blocks')
            parser.add_argument('--batch_size', type=int, default=16, help='Kích thước batch')
            parser.add_argument('--epochs', type=int, default=50, help='Số epochs huấn luyện')
            parser.add_argument('--learning_rate', type=float, default=2e-5, help='Tốc độ học tối ưu cho PCC cao')
            parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay nhẹ hơn cho PCC tốt')
            parser.add_argument('--device', type=str, default='cuda', help='Thiết bị huấn luyện (cuda/cpu)')
            parser.add_argument('--num_workers', type=int, default=0, help='Số workers cho DataLoader')
            parser.add_argument('--pin_memory', action='store_true', help='Sử dụng pin_memory cho DataLoader')
            parser.add_argument('--eval_only', action='store_true', help='Chỉ đánh giá mô hình, không huấn luyện')
            parser.add_argument('--use_amp', action='store_true', help='Sử dụng Automatic Mixed Precision')
            parser.add_argument('--use_mixup', action='store_true', help='Sử dụng kỹ thuật MixUp')
            parser.add_argument('--mixup_alpha', type=float, default=0.05, help='MixUp alpha nhẹ hơn cho PCC tốt')
            parser.add_argument('--balance_classes', action='store_true', help='Cân bằng lớp bằng WeightedRandomSampler')
            parser.add_argument('--use_swa', action='store_true', help='Sử dụng Stochastic Weight Averaging')
            parser.add_argument('--dropout', type=float, default=0.08, help='Dropout thấp hơn cho PCC tốt')
            parser.add_argument('--use_early_stopping', action='store_true', help='Sử dụng early stopping')
            parser.add_argument('--patience', type=int, default=8, help='Số epochs chờ đợi cải thiện trước khi dừng')
            
            args = parser.parse_args()
            
            # Đặt seed cho tái tạo kết quả
            set_seed(42)
            
            main()
        
        print("\n✅ Chương trình hoàn tất!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n🔍 Chi tiết lỗi:")
        traceback.print_exc()
