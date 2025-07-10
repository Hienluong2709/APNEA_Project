# train_sequence_lstm.py
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score

# Thêm thư mục gốc của dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from dataset.lazy_apnea_dataset import LazyApneaSequenceDataset
from models.convnext_lstm_lite import ConvNeXtLSTMLiteSequence

def train(model, train_loader, val_loader, device, epochs=20, lr=3e-4):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        train_losses, all_preds, all_labels = [], [], []

        for x, y, _ in train_loader:  # Dữ liệu huấn luyện không cần block_name
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)  # (B, T, num_classes)
            out = out.view(-1, out.shape[-1])
            y = y.view(-1)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_f1 = f1_score(train_labels := all_labels, train_preds := all_preds)
        train_acc = accuracy_score(train_labels, train_preds)

        # --- Validation ---
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        block_outputs = {}

        with torch.no_grad():
            for x, y, block_name in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = out.view(-1, out.shape[-1])
                y = y.view(-1)
                loss = criterion(out, y)

                val_losses.append(loss.item())

                preds = out.argmax(1).cpu().numpy()
                labels = y.cpu().numpy()
                block_name = block_name[0].replace(".npy", "")  # batch_size = 1

                if block_name not in block_outputs:
                    block_outputs[block_name] = []
                for pred, label in zip(preds, labels):
                    block_outputs[block_name].append((label, pred))

                val_preds.extend(preds)
                val_labels.extend(labels)

        val_f1 = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        # Lưu model tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/convnext_lstm_seq_best.pth")

        print(f"[Epoch {epoch+1}] "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print(f"✅ Huấn luyện hoàn tất. Best Val F1: {best_f1:.4f}")

    # Lưu output theo block vào CSV
    os.makedirs("predictions_by_block", exist_ok=True)
    for block_name, records in block_outputs.items():
        save_path = os.path.join("predictions_by_block", f"{block_name}.csv")
        with open(save_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["True Label", "Predicted Label"])
            writer.writerows(records)

    print("📁 Đã lưu dự đoán theo từng block tại thư mục predictions_by_block/")

def load_data(data_root, seq_len=10, batch_size=1):
    patients = sorted(os.listdir(data_root))
    datasets = []
    for p in patients:
        p_dir = os.path.join(data_root, p)
        if os.path.isdir(p_dir):
            try:
                ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len, return_name=True)
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Lỗi với {p}: {e}")
    full_dataset = ConcatDataset(datasets)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_loader, val_loader = load_data("data/blocks", seq_len=10, batch_size=1)
    model = ConvNeXtLSTMLiteSequence(num_classes=2)
    train(model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
