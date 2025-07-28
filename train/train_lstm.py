import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd

# Gắn thư mục project vào sys.path (đảm bảo import đúng)
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


def train(model, train_loader, val_loader, test_loader, device, epochs=3, lr=3e-4, resume_path=None):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    start_epoch = 0

    # Resume từ checkpoint nếu có
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resume from checkpoint: {resume_path}")
        model.load_state_dict(torch.load(resume_path))

    for epoch in range(start_epoch, epochs):
        model.train()
        all_preds, all_labels = [], []

        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
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

        # Lưu checkpoint tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/convnext_lstm_seq_best.pth")

        # Lưu định kỳ mỗi 2 epoch
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

        print(f"[Epoch {epoch + 1}] Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print(f"\n✅ Huấn luyện hoàn tất. Best Val F1: {best_f1:.4f}")

    # Đánh giá trên tập test
    model.load_state_dict(torch.load("checkpoints/convnext_lstm_seq_best.pth"))
    evaluate(model, test_loader, device, name="Testing")


def load_data(data_root, seq_len=5, batch_size=8):
    patients = sorted(os.listdir(data_root))
    datasets = []

    print(f"📂 Đang load dữ liệu từ {data_root}...")
    for p in patients:
        p_dir = os.path.join(data_root, p)
        if os.path.isdir(p_dir):
            try:
                print(f"📥 Loading block: {p}")
                ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len)
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def predict_and_save_csv_per_block(model, data_root, device, seq_len=5):
    model.eval()
    os.makedirs("predictions", exist_ok=True)
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
            save_path = f"predictions/{p}_preds.csv"
            df.to_csv(save_path, index=False)
            print(f"✅ Đã lưu: {save_path}")

        except Exception as e:
            print(f"⚠️ Lỗi với block {p}: {e}")


if __name__ == "__main__":
    data_path = os.path.abspath("/content/drive/MyDrive/data/blocks")

    if not os.path.exists(data_path):
        raise RuntimeError(f"❌ Không tìm thấy thư mục: {data_path}")

    os.makedirs("checkpoints", exist_ok=True)
    print("🚀 Bắt đầu huấn luyện ConvNeXt-LSTM sequence...")

    train_loader, val_loader, test_loader = load_data(data_path, seq_len=5, batch_size=8)
    model = ConvNeXtZ_LSTMLiteSequence(num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Nếu muốn resume từ checkpoint, truyền đường dẫn ở đây:
    # resume_ckpt = "checkpoints/convnext_lstm_seq_best.pth"
    # train(model, train_loader, val_loader, test_loader, device, epochs=5, resume_path=resume_ckpt)
    
    train(model, train_loader, val_loader, test_loader, device, epochs=3)

    model.load_state_dict(torch.load("checkpoints/convnext_lstm_dependent_seq_best.pth"))
    predict_and_save_csv_per_block(model, data_path, device)