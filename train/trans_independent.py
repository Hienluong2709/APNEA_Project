import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from dataset.lazy_apnea_dataset import LazyApneaSequenceDataset
from models.convnext_transformer_lite import ConvNeXtTransformerLite

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
    print(f"\n📊 {name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return acc, f1

def train(model, train_loader, val_loader, test_loader, device, epochs=5, lr=1e-4, resume_path=None):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path))
        print(f"✅ Resume từ checkpoint: {resume_path}")

    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []

        print(f"\n🔁 Epoch {epoch + 1}/{epochs} - Training...")
        for x, y in tqdm(train_loader):
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

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "checkpoints/convnext_transformer_best.pth")

        print(f"[Epoch {epoch + 1}] Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print(f"\n✅ Training hoàn tất. Best Val F1: {best_f1:.4f}")
    model.load_state_dict(torch.load("checkpoints/convnext_transformer_best.pth"))
    test_acc, test_f1 = evaluate(model, test_loader, device, name="Testing")
    print(f"\n📌 So sánh:")
    print(f"✅ Validation - F1: {best_f1:.4f}, Accuracy: {val_acc:.4f}")
    print(f"✅ Testing    - F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}")

def predict_and_save_csv_per_block(model, data_root, device, seq_len=5):
    model.eval()
    os.makedirs("predictions_trans", exist_ok=True)
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
                "label_pred": all_preds
            })
            df["correct"] = df["label_true"] == df["label_pred"]
            save_path = f"predictions_trans/{p}_preds.csv"
            df.to_csv(save_path, index=False)
            print(f"✅ Đã lưu: {save_path}")

        except Exception as e:
            print(f"⚠️ Lỗi với block {p}: {e}")

def load_data_independent(data_root, seq_len=5, batch_size=8):
    patients = sorted(os.listdir(data_root))
    num_patients = len(patients)

    train_p = patients[:int(0.8 * num_patients)]
    val_p = patients[int(0.8 * num_patients):int(0.9 * num_patients)]
    test_p = patients[int(0.9 * num_patients):]

    def make_dataset(p_list):
        datasets = []
        for p in p_list:
            p_dir = os.path.join(data_root, p)
            if os.path.isdir(p_dir):
                try:
                    ds = LazyApneaSequenceDataset(p_dir, seq_len=seq_len)
                    datasets.append(ds)
                except Exception as e:
                    print(f"⚠️ Lỗi với {p}: {e}")
        return ConcatDataset(datasets)

    train_loader = DataLoader(make_dataset(train_p), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_dataset(val_p), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(make_dataset(test_p), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_path = os.path.abspath("data/blocks")
    os.makedirs("checkpoints", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = load_data_independent(data_path, seq_len=5, batch_size=8)
    model = ConvNeXtTransformerLite(num_classes=2)
    train(model, train_loader, val_loader, test_loader, device)
    predict_and_save_csv_per_block(model, data_path, device)
