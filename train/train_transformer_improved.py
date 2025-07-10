"""
Script huấn luyện mô hình ConvNeXtTransformerLite và tính toán chỉ số AHI 
theo phương pháp Dependent Subject với các kỹ thuật tiên tiến
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import copy
import random
from torch.cuda.amp import autocast, GradScaler

# Thêm class ExponentialMovingAverage để ổn định quá trình huấn luyện
class ExponentialMovingAverage:
    """
    Implement EMA (Exponential Moving Average) để ổn định huấn luyện.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Khởi tạo shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        # Backup current parameters for restore later
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        # Restore backed up parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# Thêm class EarlyStopping để dừng huấn luyện khi không cải thiện
class EarlyStopping:
    """
    Early stopping để tránh overfitting
    """
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Kỹ thuật MixUp Loss
class MixUpLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        
    def forward(self, pred, y_a, y_b, lam):
        loss_a = self.criterion(pred, y_a)
        loss_b = self.criterion(pred, y_b)
        return lam * loss_a + (1 - lam) * loss_b

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các module
from models.convnext_transformer_lite import ConvNeXtTransformerLite
from dataset.lazy_apnea_dataset import LazyApneaSingleDataset as LazyApneaDataset
from dataset.lazy_apnea_dataset import MixUpDataset
try:
    from utils.data_splitting import dependent_subject_split
    from utils.visualization import plot_training_history, plot_confusion_matrix, plot_ahi_comparison
except ImportError:
    print("⚠️ Các module trong utils không tìm thấy, đang định nghĩa lại các hàm cần thiết...")
    
    # Định nghĩa lại hàm dependent_subject_split
    from torch.utils.data import Subset, ConcatDataset
    import random
    
    def dependent_subject_split(datasets, patient_ids, train_ratio=0.8, seed=42):
        """
        Chia dữ liệu theo phương pháp dependent subject: 
        Dữ liệu của mỗi bệnh nhân được chia thành tập train và val
        """
        random.seed(seed)
        train_subsets = []
        val_subsets = []
        
        for ds, patient_id in zip(datasets, patient_ids):
            indices = list(range(len(ds)))
            random.shuffle(indices)
            
            split_idx = int(train_ratio * len(indices))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            train_subsets.append(Subset(ds, train_indices))
            val_subsets.append(Subset(ds, val_indices))
        
        return train_subsets, val_subsets
    
    # Định nghĩa các hàm plot
    def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, 
                            metric_name='F1', save_path=None):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics, label=f'Train {metric_name}')
        plt.plot(val_metrics, label=f'Validation {metric_name}')
        plt.title(f'{metric_name} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"📊 Đã lưu biểu đồ tại: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"📊 Đã lưu confusion matrix tại: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_ahi_comparison(true_ahi, pred_ahi, patient_ids=None, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.scatter(true_ahi, pred_ahi, alpha=0.7)
        max_val = max(max(true_ahi), max(pred_ahi))
        plt.plot([0, max_val], [0, max_val], 'k--')
        plt.xlabel('AHI thực tế')
        plt.ylabel('AHI dự đoán')
        plt.title('So sánh AHI thực tế và dự đoán')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"📊 Đã lưu biểu đồ AHI tại: {save_path}")
        else:
            plt.show()
        plt.close()

# Định nghĩa các hàm liên quan đến AHI
def calculate_ahi_from_predictions(y_true, y_pred, block_duration_sec=30):
    """
    Tính AHI từ nhãn thực tế và nhãn dự đoán.
    """
    # Đảm bảo y_pred là 0 hoặc 1
    if y_pred.dtype != int:
        y_pred = (y_pred > 0.5).astype(int)

    # Tính số giờ
    total_hours = (len(y_true) * block_duration_sec) / 3600
    
    # Tính AHI
    true_apnea_count = np.sum(y_true)
    pred_apnea_count = np.sum(y_pred)
    
    true_ahi = true_apnea_count / total_hours if total_hours > 0 else 0
    pred_ahi = pred_apnea_count / total_hours if total_hours > 0 else 0
    
    # Các chỉ số đánh giá
    mae = abs(true_ahi - pred_ahi)
    rmse = np.sqrt((true_ahi - pred_ahi)**2)
    
    metrics = {
        "mae": mae,
        "rmse": rmse
    }
    
    return true_ahi, pred_ahi, metrics

def classify_osa_severity(ahi):
    """
    Phân loại mức độ nghiêm trọng của OSA dựa trên AHI
    """
    if ahi < 5:
        return "Normal"
    elif ahi < 15:
        return "Mild"
    elif ahi < 30:
        return "Moderate"
    else:
        return "Severe"

def set_seed(seed=42):
    """
    Đặt seed cho tái tạo kết quả
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, epochs=20, lr=3e-4, device='cuda', 
                patience=7, use_amp=True, use_mixup=False, mixup_alpha=0.2, 
                use_swa=True, swa_start=5, swa_freq=1):
    """
    Huấn luyện và đánh giá mô hình với nhiều kỹ thuật tiên tiến
    """
    print(f"🚀 Bắt đầu huấn luyện mô hình {model.__class__.__name__} trên {device}")
    
    # Tạo thư mục checkpoints
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Khởi tạo EMA (Exponential Moving Average)
    ema = ExponentialMovingAverage(model, decay=0.999)
    
    # Early stopping
    checkpoint_path = os.path.join(checkpoints_dir, f'{model.__class__.__name__}_best.pth')
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)
    
    # Tính class weights để xử lý imbalance
    all_labels = []
    for batch_data in train_loader:
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            _, y = batch_data
            all_labels.extend(y.numpy())
        elif isinstance(batch_data, tuple) and len(batch_data) == 4:  # MixUp returns (x, y_a, y_b, lam)
            _, y_a, _, _ = batch_data
            all_labels.extend(y_a.numpy())
    
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    class_weights = torch.tensor(
        1.0 / label_counts, 
        dtype=torch.float32, 
        device=device
    )
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Khởi tạo loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Khởi tạo MixUpLoss nếu sử dụng mixup
    if use_mixup:
        mixup_criterion = MixUpLoss(criterion)
    
    # Khởi tạo optimizer với weight decay và AdamW
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Sử dụng OneCycleLR với warm-up và cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% của epochs cho warm-up
        div_factor=25,  # lr_start = max_lr/25
        final_div_factor=1000,  # lr_end = lr_start/1000
        anneal_strategy='cos'
    )
    
    # Khởi tạo GradScaler cho Automatic Mixed Precision (AMP)
    scaler = GradScaler() if use_amp else None
    
    # Khởi tạo Stochastic Weight Averaging nếu được yêu cầu
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, anneal_strategy="linear", 
            anneal_epochs=5, swa_lr=lr/10
        )
    
    best_val_f1 = 0
    
    # Lưu lịch sử huấn luyện
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Xử lý dữ liệu MixUp
                if use_mixup and len(batch_data) == 4:
                    x, y_a, y_b, lam = batch_data
                    x = x.to(device)
                    y_a, y_b = y_a.to(device), y_b.to(device)
                    
                    # Kiểm tra dữ liệu đầu vào
                    if x.size(0) == 0 or y_a.size(0) == 0 or y_b.size(0) == 0:
                        print(f"⚠️ Bỏ qua batch {batch_idx} do rỗng")
                        continue
                else:
                    x, y = batch_data
                    
                    # Kiểm tra dữ liệu đầu vào
                    if x.size(0) == 0 or y.size(0) == 0:
                        print(f"⚠️ Bỏ qua batch {batch_idx} do rỗng")
                        continue
                    
                    x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                
                # Sử dụng AMP nếu được yêu cầu
                if use_amp:
                    with autocast():
                        pred = model(x)
                        
                        if use_mixup and len(batch_data) == 4:
                            loss = mixup_criterion(pred, y_a, y_b, lam)
                        else:
                            loss = criterion(pred, y)
                    
                    # Scale gradients và tối ưu hóa
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # Thêm gradient clipping để ổn định huấn luyện
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x)
                    
                    if use_mixup and len(batch_data) == 4:
                        loss = mixup_criterion(pred, y_a, y_b, lam)
                    else:
                        loss = criterion(pred, y)
                    
                    loss.backward()
                    
                    # Thêm gradient clipping để ổn định huấn luyện
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                # Cập nhật EMA sau mỗi bước optimizer
                ema.update()
                
                # Cập nhật learning rate với OneCycleLR
                scheduler.step()
                
                train_loss += loss.item()
                
                # Lưu dự đoán và nhãn cho tính toán metrics
                if use_mixup and len(batch_data) == 4:
                    # Với MixUp, chúng ta sử dụng nhãn gốc (y_a) cho đánh giá
                    preds = pred.argmax(1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_a.cpu().numpy())
                else:
                    preds = pred.argmax(1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.cpu().numpy())
                
                # Cập nhật thanh tiến trình
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'avg_loss': f"{train_loss/(batch_idx+1):.4f}",
                    'lr': f"{curr_lr:.6f}"
                })
                    
            except Exception as e:
                print(f"❌ Lỗi ở batch {batch_idx}: {e}")
                continue
        
        if len(all_preds) == 0:
            print("❌ Không có dữ liệu hợp lệ trong epoch này!")
            continue
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Cập nhật SWA sau khi đạt đến swa_start
        if use_swa and epoch >= swa_start and (epoch - swa_start) % swa_freq == 0:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # VALIDATE với EMA
        # Áp dụng EMA cho đánh giá
        ema.apply_shadow()
        
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_idx, (x, y) in enumerate(pbar):
                try:
                    # Kiểm tra dữ liệu đầu vào
                    if x.size(0) == 0 or y.size(0) == 0:
                        print(f"⚠️ Bỏ qua batch validation {batch_idx} do rỗng")
                        continue
                    
                    x, y = x.to(device), y.to(device)
                    
                    if use_amp:
                        with autocast():
                            pred = model(x)
                            loss = criterion(pred, y)
                    else:
                        pred = model(x)
                        loss = criterion(pred, y)
                    
                    val_loss += loss.item()
                    preds = pred.argmax(1).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.cpu().numpy())
                    
                    # Cập nhật thanh tiến trình
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        'avg_loss': f"{val_loss/(batch_idx+1):.4f}"
                    })
                    
                except Exception as e:
                    print(f"❌ Lỗi ở batch validation {batch_idx}: {e}")
                    continue
        
        # Restore lại model weights gốc
        ema.restore()
        
        if len(all_preds) == 0:
            print("❌ Không có dữ liệu validation hợp lệ trong epoch này!")
            continue
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Kiểm tra phân bố nhãn và dự đoán
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        
        print(f"\nPhân bố nhãn thực tế: {dict(zip(unique_labels, label_counts))}")
        print(f"Phân bố nhãn dự đoán: {dict(zip(unique_preds, pred_counts))}")
        
        # Tính F1 score với cảnh báo zero_division
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Lưu mô hình tốt nhất dựa trên F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'), exist_ok=True)
            best_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', f'{model.__class__.__name__}_best_f1.pth')
            torch.save(model.state_dict(), best_model_path)
            
            # Chi tiết kết quả
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            print(f"Confusion Matrix:\n{cm}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Kiểm tra early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Lưu lịch sử huấn luyện
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Finalize SWA sau khi huấn luyện kết thúc
    if use_swa and epoch >= swa_start:
        print("Finalizing SWA model...")
        # Cập nhật BatchNorm statistics cho SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        
        # Lưu SWA model
        swa_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', f'{model.__class__.__name__}_swa.pth')
        torch.save(swa_model.state_dict(), swa_model_path)
        
        # Đánh giá SWA model trên tập validation
        swa_model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = swa_model(x)
                preds = pred.argmax(1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        
        swa_val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        swa_val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"SWA Model Validation - Acc: {swa_val_acc:.4f}, F1: {swa_val_f1:.4f}")
        
        if swa_val_f1 > best_val_f1:
            print(f"SWA Model is better than best model (F1: {swa_val_f1:.4f} > {best_val_f1:.4f})")
            model.load_state_dict(swa_model.state_dict())
    
    # Vẽ biểu đồ lịch sử huấn luyện
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_history(
        train_losses, val_losses, 
        train_f1s, val_f1s,
        metric_name='F1', 
        save_path=os.path.join(results_dir, f"{model.__class__.__name__}_training_history.png")
    )
    
    # Lưu lịch sử huấn luyện
    np.savez(
        os.path.join(results_dir, f"{model.__class__.__name__}_history.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_accs=np.array(train_accs),
        val_accs=np.array(val_accs),
        train_f1s=np.array(train_f1s),
        val_f1s=np.array(val_f1s),
    )
    
    return model

def main():
    """
    Hàm chính chạy pipeline huấn luyện
    """
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình ConvNeXtTransformerLite')
    parser.add_argument('--data_dir', type=str, default='../data/blocks', help='Đường dẫn đến thư mục chứa dữ liệu')
    parser.add_argument('--batch_size', type=int, default=64, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=30, help='Số epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Số workers cho DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Thiết bị huấn luyện')
    parser.add_argument('--use_amp', action='store_true', help='Sử dụng Automatic Mixed Precision')
    parser.add_argument('--use_mixup', action='store_true', help='Sử dụng kỹ thuật MixUp')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha cho MixUp')
    parser.add_argument('--use_swa', action='store_true', help='Sử dụng Stochastic Weight Averaging')
    parser.add_argument('--use_augment', action='store_true', help='Sử dụng data augmentation')
    parser.add_argument('--balance_classes', action='store_true', help='Cân bằng các lớp')
    
    args = parser.parse_args()
    
    # Đặt seed
    set_seed(args.seed)
    
    # Đảm bảo đường dẫn dữ liệu là tuyệt đối
    if not os.path.isabs(args.data_dir):
        # Đường dẫn tương đối - chuyển thành tuyệt đối dựa trên vị trí script hiện tại
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, args.data_dir)
    else:
        data_dir = args.data_dir
    
    # Kiểm tra tồn tại của thư mục dữ liệu
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục dữ liệu: {data_dir}")
        print(f"Thư mục hiện tại: {os.getcwd()}")
        print(f"Vui lòng kiểm tra đường dẫn và thử lại.")
        return
    
    # Tạo dataset và dataloaders
    print(f"🔍 Đang tạo dataset từ {data_dir}...")
    dataset = LazyApneaDataset(
        data_dir, 
        augment=args.use_augment, 
        balance_classes=args.balance_classes
    )
    
    # Nếu sử dụng MixUp
    if args.use_mixup:
        print("🔄 Sử dụng kỹ thuật MixUp...")
        dataset = MixUpDataset(dataset, alpha=args.mixup_alpha)
    
    # Chia dataset thành train và validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"📊 Train: {len(train_dataset)} mẫu, Validation: {len(val_dataset)} mẫu")
    
    # Khởi tạo mô hình
    print(f"🧠 Khởi tạo mô hình ConvNeXtTransformerLite...")
    model = ConvNeXtTransformerLite(
        num_classes=2, 
        embed_dim=128, 
        num_heads=8, 
        num_transformer_layers=4, 
        dropout=0.3,
        dropout_path=0.1
    ).to(args.device)
    
    # Huấn luyện mô hình
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_amp=args.use_amp,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_swa=args.use_swa
    )
    
    print("✅ Huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()
