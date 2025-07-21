import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# Gắn thư mục gốc vào sys.path để import module bên ngoài
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dataset.lazy_apnea_dataset import LazyApneaSequenceDataset
from models.convnext_lstm_lite import ConvNeXtZ_LSTMLiteSequence


def evaluate(model, dataloader, device, name=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out.view(-1, out.shape[-1])
            y = y.view(-1)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"📊 {name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return acc, f1


def train_anti_overfit(model, train_loader, val_loader, test_loader, device, epochs=20, lr=5e-5):
    """
    Training với nhiều kỹ thuật chống overfitting:
    - Learning rate thấp hơn
    - Weight decay cao
    - Gradient clipping  
    - Early stopping nghiêm ngặt
    - Cosine annealing scheduler
    """
    model = model.to(device)
    
    # Optimizer với weight decay cao
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=5e-4,  # Tăng weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing chống overfit
    best_f1 = 0
    patience_counter = 0
    max_patience = 3  # Early stopping nghiêm ngặt hơn
    
    # Để theo dõi overfitting
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    print(f"🆕 Anti-overfitting training. Tổng số tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"📋 LR: {lr}, Weight decay: 5e-4, Label smoothing: 0.1")

    for epoch in range(epochs):
        # Training phase
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader))):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = out.view(-1, out.shape[-1])
            y = y.view(-1)
            loss = criterion(out, y)
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        # Tính metrics
        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds)
        
        # Validation phase
        model.eval()
        val_all_preds, val_all_labels = [], []
        val_total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = out.view(-1, out.shape[-1])
                y = y.view(-1)
                loss = criterion(out, y)
                val_total_loss += loss.item()
                
                preds = torch.argmax(out, dim=1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(y.cpu().numpy())
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_f1 = f1_score(val_all_labels, val_all_preds)
        
        # Lưu metrics để theo dõi
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        # Kiểm tra overfitting
        overfitting_gap = train_f1 - val_f1
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} (Gap: {overfitting_gap:.4f})")
        print(f"LR: {current_lr:.2e}")

        # Early stopping dựa trên validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/convnext_lstm_seq_best_anti_overfit.pth")
            print(f"✅ Saved best model with Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{max_patience}")
            
        # Cảnh báo overfitting
        if overfitting_gap > 0.15:  # Gap quá lớn
            print(f"🚨 Warning: Potential overfitting detected! Gap = {overfitting_gap:.4f}")
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

        # Lưu checkpoint định kỳ
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/anti_overfit_epoch_{epoch+1}.pth")

    print(f"\n✅ Training completed. Best Val F1: {best_f1:.4f}")
    
    # Load best model và test
    model.load_state_dict(torch.load("checkpoints/convnext_lstm_seq_best_anti_overfit.pth"))
    evaluate(model, test_loader, device, name="Final Testing")
    
    # Lưu training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'overfitting_gap': [t - v for t, v in zip(train_f1s, val_f1s)]
    })
    history_df.to_csv("training_history_anti_overfit.csv", index=False)
    print("📊 Training history saved to training_history_anti_overfit.csv")


def load_data_with_strong_augment(data_root, seq_len=5, batch_size=6):
    """
    Load data với augmentation mạnh hơn và batch size nhỏ hơn
    """
    patients = sorted([p for p in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, p))])
    train_pats, temp_pats = train_test_split(patients, test_size=0.2, random_state=42)
    val_pats, test_pats = train_test_split(temp_pats, test_size=0.5, random_state=42)

    def load_blocks(pat_list, name, use_augment=False):
        datasets = []
        print(f"\n📦 Loading {name} data...")
        for p in pat_list:
            p_dir = os.path.join(data_root, p)
            try:
                # Sử dụng augmentation mạnh cho training
                ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len, augment=use_augment)
                print(f"✅ {name} - {p} ({len(ds)} sequences) - Strong Augment: {use_augment}")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Lỗi với {p}: {e}")
        if not datasets:
            raise RuntimeError(f"❌ Không có dữ liệu trong tập {name}")
        return ConcatDataset(datasets)

    # Training với augmentation mạnh
    train_ds = load_blocks(train_pats, "Train", use_augment=True)
    val_ds = load_blocks(val_pats, "Validation", use_augment=False)
    test_ds = load_blocks(test_pats, "Test", use_augment=False)

    with open("patient_split_log_anti_overfit.txt", "w") as f:
        f.write("Train patients:\n" + "\n".join(train_pats) + "\n\n")
        f.write("Validation patients:\n" + "\n".join(val_pats) + "\n\n")
        f.write("Test patients:\n" + "\n".join(test_pats) + "\n")

    print("📁 Ghi danh sách bệnh nhân vào: patient_split_log_anti_overfit.txt")

    # Batch size nhỏ hơn để regularization
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_path = os.path.abspath("data/blocks")
    if not os.path.exists(data_path):
        raise RuntimeError(f"❌ Không tìm thấy thư mục: {data_path}")

    os.makedirs("checkpoints", exist_ok=True)
    print("🚀 Anti-Overfitting Training for ConvNeXt-LSTM sequence...")

    # Load data với augmentation mạnh
    train_loader, val_loader, test_loader = load_data_with_strong_augment(data_path, seq_len=5, batch_size=6)
    
    # Model với dropout cao hơn
    model = ConvNeXtZ_LSTMLiteSequence(num_classes=2, dropout=0.6)  # Dropout cao
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training với các kỹ thuật chống overfitting
    train_anti_overfit(model, train_loader, val_loader, test_loader, device, epochs=20, lr=5e-5)
