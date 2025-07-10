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
import traceback
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
                print(f'Bộ đếm Early Stopping: {self.counter} trên {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss giảm ({self.val_loss_min:.6f} --> {val_loss:.6f}). Đang lưu mô hình...')
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
        plt.title('Loss theo epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics, label=f'Train {metric_name}')
        plt.plot(val_metrics, label=f'Validation {metric_name}')
        plt.title(f'{metric_name} theo epochs')
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
        plt.title('Ma trận nhầm lẫn')
        plt.ylabel('Nhãn thực tế')
        plt.xlabel('Nhãn dự đoán')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"📊 Đã lưu ma trận nhầm lẫn tại: {save_path}")
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
        return "Bình thường"
    elif ahi < 15:
        return "Nhẹ"
    elif ahi < 30:
        return "Trung bình"
    else:
        return "Nặng"

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
    
    Tham số:
        model: Mô hình cần huấn luyện
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        epochs: Số epoch huấn luyện
        lr: Tốc độ học
        device: Thiết bị huấn luyện (cuda/cpu)
        patience: Số epoch chờ trước khi dừng sớm
        use_amp: Sử dụng Automatic Mixed Precision
        use_mixup: Sử dụng kỹ thuật MixUp
        mixup_alpha: Hệ số alpha cho MixUp
        use_swa: Sử dụng Stochastic Weight Averaging
        swa_start: Epoch bắt đầu sử dụng SWA
        swa_freq: Tần suất cập nhật SWA
        
    Trả về:
        model: Mô hình đã huấn luyện
        best_val_f1: Giá trị F1 tốt nhất đạt được
    """
    print(f"🚀 Bắt đầu huấn luyện mô hình {model.__class__.__name__} trên {device}")
    
    # Khởi tạo EMA (Exponential Moving Average)
    ema = ExponentialMovingAverage(model, decay=0.999)
    
    # Early stopping
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', f'{model.__class__.__name__}_best.pth')
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)
    
    # Tính class weights để xử lý imbalance
    all_labels = []
    for _, y in train_loader:
        if isinstance(y, tuple):  # MixUp returns (y_a, y_b, lam)
            all_labels.extend(y[0].numpy())
        else:
            all_labels.extend(y.numpy())
    
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    class_weights = torch.tensor(
        1.0 / label_counts, 
        dtype=torch.float32, 
        device=device
    )
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"Trọng số lớp: {class_weights.cpu().numpy()}")
    
    # Khởi tạo loss function với trọng số mạnh hơn cho lớp thiểu số
    # Tăng gấp đôi trọng số cho lớp Ngưng thở (class 1)
    enhanced_weights = class_weights.clone()
    if len(enhanced_weights) > 1:
        enhanced_weights[1] = enhanced_weights[1] * 2.0
        # Chuẩn hóa lại
        enhanced_weights = enhanced_weights / enhanced_weights.sum() * len(enhanced_weights)
        print(f"Trọng số lớp mạnh hơn: {enhanced_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=enhanced_weights)
    
    # Khởi tạo MixUpLoss nếu sử dụng mixup
    if use_mixup:
        mixup_criterion = MixUpLoss(criterion)
    
    # Khởi tạo optimizer với weight decay và AdamW
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Thêm focal loss para để giải quyết vấn đề mất cân bằng
    focal_gamma = 2.0  # Tham số gamma cho focal loss (tăng trọng số cho các mẫu khó phân loại)
    print(f"Sử dụng Focal Loss với gamma={focal_gamma}")
    
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
                            # Thêm focal loss component
                            probs = torch.softmax(pred, dim=1)
                            pt = probs.gather(1, y.view(-1, 1)).squeeze(1)
                            focal_weight = (1 - pt).pow(focal_gamma)
                            loss = criterion(pred, y)
                            loss = (focal_weight * loss).mean()
                    
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
                        # Thêm focal loss component
                        probs = torch.softmax(pred, dim=1)
                        pt = probs.gather(1, y.view(-1, 1)).squeeze(1)
                        focal_weight = (1 - pt).pow(focal_gamma)
                        loss = criterion(pred, y)
                        loss = (focal_weight * loss).mean()
                    
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
        if len(np.unique(all_preds)) == 1:
            print(f"⚠️ Cảnh báo: Mô hình chỉ dự đoán một lớp ({np.unique(all_preds)[0]}) trong validation")
            val_f1 = 0.0 if np.unique(all_preds)[0] != np.unique(all_labels)[0] else 1.0
        else:
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
            print(f"Ma trận nhầm lẫn:\n{cm}")
            print(f"Độ chính xác: {precision:.4f}, Độ phủ: {recall:.4f}")
        
        # Kiểm tra early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping được kích hoạt sau epoch {epoch+1}")
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
        print("Đang hoàn thiện mô hình SWA...")
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
        
        print(f"Đánh giá mô hình SWA - Acc: {swa_val_acc:.4f}, F1: {swa_val_f1:.4f}")
        
        if swa_val_f1 > best_val_f1:
            print(f"Mô hình SWA tốt hơn mô hình tốt nhất (F1: {swa_val_f1:.4f} > {best_val_f1:.4f})")
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
    
    return model, best_val_f1

def evaluate_ahi(model, patient_datasets, device='cuda'):
    """
    Đánh giá mô hình trên từng bệnh nhân và tính AHI
    """
    true_ahis = []
    pred_ahis = []
    patient_ids = []
    ahi_results = {}
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    for patient_id, dataset in patient_datasets.items():
        print(f"⏳ Đánh giá bệnh nhân {patient_id}...")
        
        # DataLoader cho bệnh nhân
        patient_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Dự đoán
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(patient_loader, desc=f"Dự đoán cho bệnh nhân {patient_id}"):
                try:
                    x = x.to(device)
                    outputs = model(x)
                    preds = outputs.argmax(1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())
                except Exception as e:
                    print(f"❌ Lỗi khi dự đoán: {e}")
                    continue
        
        # Tính AHI
        if len(all_preds) > 0:
            true_ahi, pred_ahi, metrics = calculate_ahi_from_predictions(
                np.array(all_labels), np.array(all_preds)
            )
            
            true_severity = classify_osa_severity(true_ahi)
            pred_severity = classify_osa_severity(pred_ahi)
            
            print(f"  AHI thực tế: {true_ahi:.2f} ({true_severity})")
            print(f"  AHI dự đoán: {pred_ahi:.2f} ({pred_severity})")
            print(f"  MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
            
            true_ahis.append(true_ahi)
            pred_ahis.append(pred_ahi)
            patient_ids.append(patient_id)
            
            ahi_results[patient_id] = {
                'true_ahi': true_ahi,
                'pred_ahi': pred_ahi,
                'true_severity': true_severity,
                'pred_severity': pred_severity,
                'metrics': metrics
            }
    
    # Lưu kết quả
    np.savez(
        os.path.join(results_dir, 'ahi_evaluation.npz'),
        patient_ids=np.array(patient_ids),
        true_ahis=np.array(true_ahis),
        pred_ahis=np.array(pred_ahis)
    )
    
    # Vẽ biểu đồ so sánh AHI
    if len(true_ahis) > 0:
        plot_ahi_comparison(
            true_ahis, 
            pred_ahis,
            patient_ids=patient_ids,
            save_path=os.path.join(results_dir, 'ahi_comparison.png')
        )
        
        # Tạo confusion matrix cho phân loại OSA
        true_severities = [classify_osa_severity(ahi) for ahi in true_ahis]
        pred_severities = [classify_osa_severity(ahi) for ahi in pred_ahis]
        
        severity_classes = ['Bình thường', 'Nhẹ', 'Trung bình', 'Nặng']
        severity_mapping = {'Bình thường': 0, 'Nhẹ': 1, 'Trung bình': 2, 'Nặng': 3}
        
        true_classes = [severity_mapping[s] for s in true_severities]
        pred_classes = [severity_mapping[s] for s in pred_severities]
        
        cm = confusion_matrix(true_classes, pred_classes, labels=range(len(severity_classes)))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=severity_classes, yticklabels=severity_classes)
        plt.title('Phân loại mức độ nghiêm trọng OSA')
        plt.ylabel('Mức độ thực tế')
        plt.xlabel('Mức độ dự đoán')
        plt.savefig(os.path.join(results_dir, 'osa_severity_confusion.png'))
        plt.close()
    
    return true_ahis, pred_ahis, patient_ids, ahi_results

def main():
    """
    Hàm chính thực hiện quá trình huấn luyện và đánh giá mô hình
    """
    # Xử lý tham số dòng lệnh từ biến args toàn cục
    # Đã được xử lý ở phần if __name__ == "__main__"
    
    # Đường dẫn dữ liệu
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Nếu data_dir được cung cấp, dùng nó để xác định đường dẫn
    if args.data_dir:
        # Xử lý đường dẫn tương đối hoặc tuyệt đối
        if os.path.isabs(args.data_dir):
            data_dir = args.data_dir
        else:
            # Xử lý đường dẫn tương đối từ vị trí hiện tại
            data_dir = os.path.abspath(os.path.join(os.getcwd(), args.data_dir))
            
            # Nếu vẫn không tìm thấy, thử từ thư mục gốc của dự án
            if not os.path.exists(data_dir):
                data_dir = os.path.abspath(os.path.join(project_dir, args.data_dir))
    else:
        # Đường dẫn mặc định
        data_dir = os.path.join(project_dir, "data", "blocks")
    
    print(f"🔍 Đường dẫn dữ liệu: {data_dir}")
    
    # Kiểm tra tồn tại của thư mục dữ liệu
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục blocks tại {data_dir}")
        print(f"⚠️ Thư mục hiện tại: {os.getcwd()}")
        print("⚠️ Vui lòng chạy build_dataset.py trước hoặc kiểm tra lại đường dẫn")
        return
    
    # Tổng hợp dữ liệu từ tất cả các bệnh nhân
    try:
        patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        print(f"📊 Đang tải dữ liệu từ {len(patient_dirs)} bệnh nhân")
        
        datasets = []
        patient_ids = []
        patient_datasets = {}
        
        for p_dir in patient_dirs:
            try:
                patient_id = os.path.basename(p_dir)
                ds = LazyApneaDataset(p_dir)
                if len(ds) > 0:
                    datasets.append(ds)
                    patient_ids.append(patient_id)
                    patient_datasets[patient_id] = ds
                    print(f"✅ Đã tải {len(ds)} mẫu từ {patient_id}")
            except Exception as e:
                print(f"❌ Lỗi khi tải dữ liệu từ {p_dir}: {e}")
        
        if not datasets:
            print("❌ Không có dữ liệu để huấn luyện. Vui lòng chạy build_dataset.py trước.")
            return
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        return
    
    # Chia dữ liệu theo phương pháp dependent subject
    print("\n🔀 Đang chia dữ liệu theo phương pháp dependent subject...")
    try:
        train_datasets, val_datasets = dependent_subject_split(
            datasets, patient_ids, train_ratio=0.8, seed=42
        )
        
        # Tạo combined datasets
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        
        print(f"📈 Tổng số mẫu: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        # Kiểm tra tính hiệu quả của phương pháp dependent subject
        print("\n🔍 Kiểm tra phương pháp dependent subject:")
        for i, (p_id, p_ds) in enumerate(zip(patient_ids, datasets)):
            train_count = len(train_datasets[i])
            val_count = len(val_datasets[i])
            total = len(p_ds)
            print(f"  {p_id}: {train_count}/{total} ({train_count/total*100:.1f}%) train, "
                  f"{val_count}/{total} ({val_count/total*100:.1f}%) validation")
    except Exception as e:
        print(f"❌ Lỗi khi chia dữ liệu: {e}")
        return
    
    # Kiểm tra phân bố nhãn trong tập train
    print("\n📊 Kiểm tra phân bố nhãn:")
    train_labels = []
    val_labels = []
    
    # Lấy tất cả các nhãn từ tập train
    for ds in train_datasets:
        for i in range(len(ds)):
            _, label = ds[i]
            train_labels.append(label)
    
    # Lấy tất cả các nhãn từ tập validation
    for ds in val_datasets:
        for i in range(len(ds)):
            _, label = ds[i]
            val_labels.append(label)
    
    # Tính toán class weights cho loss
    train_labels_np = np.array(train_labels)
    num_samples = len(train_labels_np)
    num_classes = 2  # Bình thường và Ngưng thở
    
    class_counts = [np.sum(train_labels_np == i) for i in range(num_classes)]
    print(f"  Tập train: Bình thường: {class_counts[0]}, Ngưng thở: {class_counts[1]}")
    
    # Tính trọng số các lớp cho tập train - ưu tiên mạnh hơn cho lớp thiểu số
    # Sử dụng công thức cân bằng mạnh hơn
    total = sum(class_counts)
    class_weights = [total / (num_classes * count) for count in class_counts]
    print(f"  Trọng số: Bình thường: {class_weights[0]:.2f}, Ngưng thở: {class_weights[1]:.2f}")
    
    # Tính phân bố nhãn cho tập validation
    val_labels_np = np.array(val_labels)
    val_class_counts = [np.sum(val_labels_np == i) for i in range(num_classes)]
    print(f"  Tập validation: Bình thường: {val_class_counts[0]}, Ngưng thở: {val_class_counts[1]}")
    
    # Kiểm tra phân bố lớp trong tập huấn luyện và validation
    print("\n📊 Kiểm tra phân bố lớp (class distribution):")
    
    # Tập huấn luyện
    train_labels = []
    for ds in train_datasets:
        for _, y in ds:
            train_labels.append(y.item())
    
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    print(f"  Train: {dict(zip(train_unique, train_counts))}")
    
    # Tập validation
    val_labels = []
    for ds in val_datasets:
        for _, y in ds:
            val_labels.append(y.item())
    
    val_unique, val_counts = np.unique(val_labels, return_counts=True)
    print(f"  Validation: {dict(zip(val_unique, val_counts))}")
    
    # Tính tỷ lệ mất cân bằng và hiển thị
    if len(train_unique) > 1:
        imbalance_ratio = max(train_counts) / min(train_counts)
        print(f"  Tỷ lệ mất cân bằng: {imbalance_ratio:.2f}")
        
        # Tính trọng số nghịch đảo cho từng lớp
        class_weights = len(train_labels) / (len(train_unique) * train_counts)
        print(f"  Trọng số lớp: {dict(zip(train_unique, class_weights))}")
    
    # Tạo DataLoader với các tùy chọn tiên tiến
    try:
        # DataLoader cho tập huấn luyện
        if args.balance_classes:
            print("\n⚖️ Đang cân bằng dữ liệu cho tập train...")
            samples_weight = np.array([class_weights[t] for t in train_labels])
            samples_weight = torch.from_numpy(samples_weight).float()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            # Sử dụng MixUp dataset nếu cần
            if args.use_mixup:
                print("\n🔄 Đang áp dụng kỹ thuật MixUp...")
                try:
                    mixup_dataset = MixUpDataset(train_dataset, alpha=args.mixup_alpha)
                    train_loader = DataLoader(
                        mixup_dataset, 
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                    )
                except Exception as e:
                    print(f"⚠️ Lỗi khi tạo MixUp dataset: {e}. Sử dụng dataset thông thường.")
                    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                    )
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True
                )
        
        # DataLoader cho tập validation
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    except Exception as e:
        print(f"❌ Lỗi khi tạo DataLoader: {e}")
        return
    
    # Chọn thiết bị
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\n🖥️ Sử dụng thiết bị: {device}")
    
    # Tạo class weights tensor cho loss function
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Khởi tạo mô hình
    try:
        model = ConvNeXtTransformerLite(num_classes=2).to(device)
        print(f"✅ Khởi tạo mô hình {model.__class__.__name__} thành công")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo mô hình: {e}")
        return
    
    # Huấn luyện hoặc tải mô hình đã huấn luyện
    try:
        if not args.eval_only:
            # Tạo thư mục checkpoints nếu chưa tồn tại
            checkpoint_dir = os.path.join(project_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            print(f"\n🚀 Bắt đầu huấn luyện mô hình với {args.epochs} epochs...")
            model, best_val_f1 = train_model(
                model, 
                train_loader, 
                val_loader, 
                epochs=args.epochs, 
                lr=args.learning_rate,
                device=device,
                use_amp=args.use_amp,
                use_mixup=args.use_mixup,
                mixup_alpha=args.mixup_alpha,
                use_swa=args.use_swa
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
    
    # Đánh giá trên tập validation và vẽ confusion matrix
    try:
        print("\n📊 Đánh giá mô hình trên tập validation...")
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc="Đánh giá trên tập validation")):
                try:
                    if x.size(0) == 0:
                        continue
                    x = x.to(device)
                    outputs = model(x)
                    preds = outputs.argmax(1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.numpy())
                except Exception as e:
                    print(f"❌ Lỗi khi đánh giá batch {batch_idx}: {e}")            # Tính các chỉ số đánh giá
            accuracy = accuracy_score(all_labels, all_preds)
            # Xử lý trường hợp khi mô hình chỉ dự đoán một lớp
            if len(np.unique(all_preds)) == 1:
                print(f"⚠️ Cảnh báo: Mô hình chỉ dự đoán một lớp ({np.unique(all_preds)[0]})")
                f1 = 0.0 if np.unique(all_preds)[0] != np.unique(all_labels)[0] else 1.0
                precision = 0.0 if np.unique(all_preds)[0] != np.unique(all_labels)[0] else 1.0
                recall = 0.0 if np.unique(all_preds)[0] != np.unique(all_labels)[0] else 1.0
            else:
                f1 = f1_score(all_labels, all_preds, average='weighted')
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
            
            print(f"\n📈 Kết quả đánh giá trên tập validation:")
            print(f"  Độ chính xác: {accuracy:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
        
        # Vẽ confusion matrix
        results_dir = os.path.join(project_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        plot_confusion_matrix(
            all_labels, 
            all_preds,
            classes=['Bình thường', 'Ngưng thở'], 
            save_path=os.path.join(results_dir, f"{model.__class__.__name__}_confusion_matrix.png")
        )
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá mô hình: {e}")
        return
    
    # Tính AHI và đánh giá
    try:
        print("\n📊 Đánh giá AHI trên từng bệnh nhân...")
        true_ahis, pred_ahis, patient_ids, ahi_results = evaluate_ahi(
            model, patient_datasets, device=device
        )
        
        if len(true_ahis) > 0:
            # Tính các chỉ số đánh giá tổng hợp
            mae = np.mean(np.abs(np.array(true_ahis) - np.array(pred_ahis)))
            rmse = np.sqrt(np.mean((np.array(true_ahis) - np.array(pred_ahis))**2))
            correlation = np.corrcoef(true_ahis, pred_ahis)[0, 1]
            
            print(f"\n📈 Kết quả đánh giá AHI tổng hợp:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  Correlation: {correlation:.2f}")
            
            # Đánh giá phân loại mức độ nghiêm trọng
            true_severities = [classify_osa_severity(ahi) for ahi in true_ahis]
            pred_severities = [classify_osa_severity(ahi) for ahi in pred_ahis]
            
            # Tính accuracy của phân loại mức độ
            severity_accuracy = sum(t == p for t, p in zip(true_severities, pred_severities)) / len(true_severities)
            print(f"  Độ chính xác phân loại mức độ: {severity_accuracy:.2f}")
            
            # Thống kê chênh lệch
            severity_diff = {}
            for ts, ps in zip(true_severities, pred_severities):
                if ts != ps:
                    key = f"{ts} -> {ps}"
                    severity_diff[key] = severity_diff.get(key, 0) + 1
            
            if severity_diff:
                print("\n⚠️ Chênh lệch phân loại mức độ:")
                for diff, count in severity_diff.items():
                    print(f"  {diff}: {count} bệnh nhân")
        
        print("\n✅ Hoàn tất đánh giá mô hình!")
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá AHI: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Import các module cần thiết để hiển thị traceback chi tiết khi có lỗi
    import traceback
    
    try:
        print("\n🚀 Bắt đầu quá trình huấn luyện và đánh giá mô hình ConvNeXtTransformerLite...")
        parser = argparse.ArgumentParser(description='Huấn luyện mô hình ConvNeXtTransformerLite cho phát hiện ngưng thở khi ngủ')
        parser.add_argument('--data_dir', type=str, help='Đường dẫn đến thư mục dữ liệu blocks')
        parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
        parser.add_argument('--epochs', type=int, default=20, help='Số epochs huấn luyện')
        parser.add_argument('--learning_rate', type=float, default=1e-3, help='Tốc độ học ban đầu')
        parser.add_argument('--device', type=str, default='cuda', help='Thiết bị huấn luyện (cuda/cpu)')
        parser.add_argument('--num_workers', type=int, default=4, help='Số workers cho DataLoader')
        parser.add_argument('--eval_only', action='store_true', help='Chỉ đánh giá mô hình, không huấn luyện')
        parser.add_argument('--use_amp', action='store_true', help='Sử dụng Automatic Mixed Precision')
        parser.add_argument('--use_mixup', action='store_true', help='Sử dụng kỹ thuật MixUp')
        parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Hệ số alpha cho MixUp')
        parser.add_argument('--balance_classes', action='store_true', help='Cân bằng lớp bằng WeightedRandomSampler')
        parser.add_argument('--use_swa', action='store_true', help='Sử dụng Stochastic Weight Averaging')
        
        args = parser.parse_args()
        
        # Đặt seed cho tái tạo kết quả
        set_seed(42)
        
        main()
        print("\n✅ Chương trình hoàn tất!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n🔍 Chi tiết lỗi:")
        traceback.print_exc()
