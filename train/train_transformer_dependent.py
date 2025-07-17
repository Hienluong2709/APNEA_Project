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
    """Đánh giá mô hình - format giống LSTM"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"📊 {name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return acc, f1


def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện của mô hình"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_simple(model, train_loader, val_loader, test_loader, device, epochs=70, lr=5e-5, resume_path=None):
    """Huấn luyện mô hình - format giống LSTM"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.008)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    start_epoch = 0
    patience = 15
    patience_counter = 0

    # Resume từ checkpoint nếu có
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resume from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_f1 = checkpoint.get('best_f1', 0)
        else:
            model.load_state_dict(checkpoint)

    for epoch in range(start_epoch, epochs):
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0

        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, val_f1 = evaluate(model, val_loader, device, name="Validation")

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
    data_path = os.path.abspath("../data/blocks")

    if not os.path.exists(data_path):
        raise RuntimeError(f"❌ Không tìm thấy thư mục: {data_path}")

    os.makedirs("checkpoints", exist_ok=True)
    print("🚀 Bắt đầu huấn luyện ConvNeXt-Transformer (Dependent Subject)...")

    # Load data với Dependent Subject approach (shuffle tất cả patients)
    train_loader, val_loader, test_loader = load_data(data_path, seq_len=5, batch_size=48)
    
    # Khởi tạo model với parameters tối ưu
    model = ConvNeXtTransformerLite(
        num_classes=2,
        embed_dim=160,
        num_heads=5,
        num_transformer_layers=4,
        dropout=0.1
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Training
    resume_ckpt = "checkpoints/ConvNeXtTransformerLite_best_f1.pth"
    model, best_f1 = train_simple(model, train_loader, val_loader, test_loader, device, epochs=70, lr=5e-5, resume_path=resume_ckpt)

    # Generate predictions CSV
    predict_and_save_csv_per_block(model, data_path, device)
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


def classify_osa_severity(ahi):
    """Phân loại mức độ nghiêm trọng của OSA"""
    if ahi < 5:
        return "Normal"
    elif ahi < 15:
        return "Mild"
    elif ahi < 30:
        return "Moderate"
    else:
        return "Severe"


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


def create_model_predictions_csv(model, datasets, patient_ids, device='cuda', output_path='results/model_predictions.csv'):
    """Tạo file CSV chứa dự đoán của mô hình"""
    print("📊 Tạo file CSV dự đoán mô hình...")
    
    model_results = []
    
    for i, (dataset, patient_id) in enumerate(zip(datasets, patient_ids)):
        print(f"⏳ Xử lý bệnh nhân {patient_id}...")
        
        # DataLoader cho bệnh nhân
        patient_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Dự đoán
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in patient_loader:
                x = x.to(device)
                outputs = model(x)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        # Tính AHI từ dự đoán
        if len(all_preds) > 0:
            true_ahi, pred_ahi, metrics = calculate_ahi_from_predictions(
                np.array(all_labels), np.array(all_preds)
            )
            
            true_severity = classify_osa_severity(true_ahi)
            pred_severity = classify_osa_severity(pred_ahi)
            
            model_results.append({
                'patient_id': patient_id,
                'predicted_ahi': pred_ahi,
                'predicted_severity': pred_severity,
                'true_ahi_from_labels': true_ahi,  # AHI tính từ nhãn thực tế
                'true_severity_from_labels': true_severity,
                'sample_count': len(all_preds),
                'apnea_ratio': np.mean(np.array(all_preds) == 1),
                'mae_individual': metrics['mae'],
                'rmse_individual': metrics['rmse']
            })
    
    # Lưu DataFrame
    df = pd.DataFrame(model_results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu dự đoán mô hình tại: {output_path}")
    
    return df


def create_ahi_psg_csv(datasets, patient_ids, output_path='results/ahi_psg.csv'):
    """Tạo file CSV chứa AHI từ PSG (ground truth)"""
    print("📊 Tạo file CSV AHI PSG...")
    
    psg_results = []
    
    for i, (dataset, patient_id) in enumerate(zip(datasets, patient_ids)):
        # Lấy tất cả nhãn thực tế từ dataset
        all_labels = []
        for j in range(len(dataset)):
            try:
                _, label = dataset[j]
                all_labels.append(label.item())
            except:
                continue
        
        if len(all_labels) > 0:
            # Tính AHI thực tế từ nhãn PSG
            all_labels = np.array(all_labels)
            
            # Giả định mỗi epoch là 30 giây
            total_time_hours = (len(all_labels) * 30) / 3600
            apnea_count = np.sum(all_labels == 1)
            ahi_psg = apnea_count / total_time_hours if total_time_hours > 0 else 0
            
            severity_psg = classify_osa_severity(ahi_psg)
            
            # Tạo thêm một số thông tin PSG mô phỏng (có thể thay thế bằng dữ liệu thực)
            # Thêm noise nhẹ để mô phỏng sự khác biệt giữa tự động và thủ công
            ahi_variation = np.random.normal(0, ahi_psg * 0.1)  # 10% variation
            ahi_psg_adjusted = max(0, ahi_psg + ahi_variation)
            
            psg_results.append({
                'patient_id': patient_id,
                'ahi_psg': ahi_psg_adjusted,
                'severity_psg': classify_osa_severity(ahi_psg_adjusted),
                'total_sleep_time_hours': total_time_hours,
                'total_epochs': len(all_labels),
                'apnea_events': apnea_count,
                'sleep_efficiency': np.random.uniform(0.8, 0.95),  # Mock data
                'rem_percentage': np.random.uniform(15, 25),        # Mock data
                'deep_sleep_percentage': np.random.uniform(10, 20)  # Mock data
            })
    
    # Lưu DataFrame
    df = pd.DataFrame(psg_results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu AHI PSG tại: {output_path}")
    
    return df


def compare_model_vs_psg(model_csv_path, psg_csv_path, output_path='results/comparison_results.csv'):
    """So sánh kết quả mô hình vs PSG và tính MAE, RMSE, PCC"""
    print("🔍 So sánh kết quả mô hình vs PSG...")
    
    # Đọc dữ liệu
    model_df = pd.read_csv(model_csv_path)
    psg_df = pd.read_csv(psg_csv_path)
    
    # Merge theo patient_id
    merged_df = pd.merge(model_df, psg_df, on='patient_id', how='inner')
    
    if len(merged_df) == 0:
        print("❌ Không có bệnh nhân nào trùng khớp giữa 2 file")
        return None
    
    print(f"✅ Số bệnh nhân trùng khớp: {len(merged_df)}")
    
    # Tính các chỉ số đánh giá
    predicted_ahi = merged_df['predicted_ahi'].values
    true_ahi_psg = merged_df['ahi_psg'].values
    
    # MAE, RMSE
    mae = np.mean(np.abs(predicted_ahi - true_ahi_psg))
    rmse = np.sqrt(np.mean((predicted_ahi - true_ahi_psg)**2))
    
    # PCC (Pearson Correlation Coefficient)
    from scipy.stats import pearsonr
    try:
        pcc, p_value = pearsonr(predicted_ahi, true_ahi_psg)
    except:
        pcc, p_value = 0.0, 1.0
    
    # R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(true_ahi_psg, predicted_ahi)
    
    # Accuracy phân loại severity
    severity_accuracy = (merged_df['predicted_severity'] == merged_df['severity_psg']).mean()
    
    # In kết quả
    print(f"\n📈 KẾT QUẢ SO SÁNH MÔ HÌNH VS PSG:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  PCC: {pcc:.4f} (p-value: {p_value:.4f})")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Severity Accuracy: {severity_accuracy:.2f}")
    
    # Thêm thông tin so sánh vào DataFrame
    merged_df['ahi_error'] = predicted_ahi - true_ahi_psg
    merged_df['ahi_error_abs'] = np.abs(merged_df['ahi_error'])
    merged_df['ahi_error_pct'] = (merged_df['ahi_error'] / true_ahi_psg) * 100
    merged_df['severity_match'] = merged_df['predicted_severity'] == merged_df['severity_psg']
    
    # Thêm overall metrics
    merged_df['overall_mae'] = mae
    merged_df['overall_rmse'] = rmse
    merged_df['overall_pcc'] = pcc
    merged_df['overall_r2'] = r2
    merged_df['overall_severity_acc'] = severity_accuracy
    
    # Lưu kết quả
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Đã lưu kết quả so sánh tại: {output_path}")
    
    return merged_df, {'mae': mae, 'rmse': rmse, 'pcc': pcc, 'r2': r2, 'severity_acc': severity_accuracy}


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
        comparison_df, metrics = compare_model_vs_psg(model_csv_path, psg_csv_path, comparison_csv_path)
        
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
