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
PendingDeprecationWarning
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


def train(model, train_loader, val_loader, test_loader, device, epochs=5, lr=1e-4):
    model = model.to(device)
    # Thêm weight decay để regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler để giảm learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    patience_counter = 0
    max_patience = 5  # Early stopping

    print(f"🆕 Bắt đầu huấn luyện mô hình từ đầu. Tổng số tham số: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")

        for x, y in tqdm(train_loader, total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            out = out.view(-1, out.shape[-1])
            y = y.view(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_f1 = f1_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, all_preds)
        val_acc, val_f1 = evaluate(model, val_loader, device, name="Validation")

        # Early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/convnext_lstm_seq_best.pth")
            print(f"✅ Saved best model with Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{max_patience}")
            
        # Step scheduler
        scheduler.step(val_f1)
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

        print(f"[Epoch {epoch + 1}] Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print(f"\n✅ Huấn luyện hoàn tất. Best Val F1: {best_f1:.4f}")
    model.load_state_dict(torch.load("checkpoints/convnext_lstm_seq_best.pth"))
    evaluate(model, test_loader, device, name="Testing")


def load_data_independent(data_root, seq_len=5, batch_size=8):
    patients = sorted([p for p in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, p))])
    train_pats, temp_pats = train_test_split(patients, test_size=0.2, random_state=42)
    val_pats, test_pats = train_test_split(temp_pats, test_size=0.5, random_state=42)

    def load_blocks(pat_list, name, use_augment=False):
        datasets = []
        print(f"\n📦 Loading {name} data...")
        for p in pat_list:
            p_dir = os.path.join(data_root, p)
            try:
                # Chỉ sử dụng augmentation cho training data
                ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len, augment=use_augment)
                print(f"✅ {name} - {p} ({len(ds)} sequences) - Augment: {use_augment}")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Lỗi với {p}: {e}")
        if not datasets:
            raise RuntimeError(f"❌ Không có dữ liệu trong tập {name}")
        return ConcatDataset(datasets)

    # Chỉ train data mới dùng augmentation
    train_ds = load_blocks(train_pats, "Train", use_augment=True)
    val_ds = load_blocks(val_pats, "Validation", use_augment=False)
    test_ds = load_blocks(test_pats, "Test", use_augment=False)

    with open("patient_split_log.txt", "w") as f:
        f.write("Train patients:\n" + "\n".join(train_pats) + "\n\n")
        f.write("Validation patients:\n" + "\n".join(val_pats) + "\n\n")
        f.write("Test patients:\n" + "\n".join(test_pats) + "\n")

    print("📁 Ghi danh sách bệnh nhân vào: patient_split_log.txt")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def predict_and_save_csv_per_block(model, data_root, device, seq_len=5):
    model.eval()
    os.makedirs("prediction", exist_ok=True)
    patients = sorted(os.listdir(data_root))

    print(f"\n🧪 Lưu dự đoán nhị phân dưới dạng CSV cho từng block...")
    for p in patients:
        p_dir = os.path.join(data_root, p)
        if not os.path.isdir(p_dir):
            continue
        try:
            ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len)
            loader = DataLoader(ds, batch_size=8, shuffle=False)
            all_preds, all_labels = [], []

            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    out = model(x)
                    out = out.view(-1, out.shape[-1])
                    preds = torch.argmax(out, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.view(-1).cpu().numpy())

            df = pd.DataFrame({
                "label_true": all_labels,
                "label_pred": all_preds,
            })
            df["correct"] = df["label_true"] == df["label_pred"]
            save_path = f"prediction/{p}_preds.csv"
            df.to_csv(save_path, index=False)
            print(f"✅ Đã lưu: {save_path}")

        except Exception as e:
            print(f"⚠️ Lỗi với block {p}: {e}")


if __name__ == "__main__":
    data_path = os.path.abspath("data/blocks")
    if not os.path.exists(data_path):
        raise RuntimeError(f"❌ Không tìm thấy thư mục: {data_path}")

    os.makedirs("checkpoints", exist_ok=True)
    print("🚀 Bắt đầu huấn luyện ConvNeXt-LSTM sequence...")

    train_loader, val_loader, test_loader = load_data_independent(data_path, seq_len=5, batch_size=8)
    model = ConvNeXtZ_LSTMLiteSequence(num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ✅ Không load checkpoint (huấn luyện lại từ đầu)
    train(model, train_loader, val_loader, test_loader, device, epochs=5, lr=1e-4)

    predict_and_save_csv_per_block(model, data_path, device)
12 