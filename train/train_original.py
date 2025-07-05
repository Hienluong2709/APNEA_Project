import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.convnext_lstm import ConvNeXtLSTM
from dataset.lazy_apnea_dataset import PatientBlockDataset

# Cấu hình
BLOCKS_DIR = "data/blocks"
print("✅ Đường dẫn BLOCKS_DIR:", os.path.abspath(BLOCKS_DIR))
print("📁 Danh sách thư mục/file trong BLOCKS_DIR:", os.listdir(BLOCKS_DIR))
SEQ_LEN = 5
BATCH_SIZE = 8
EPOCHS = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load theo từng bệnh nhân
all_datasets = []
patient_dirs = [os.path.join(BLOCKS_DIR, d) for d in os.listdir(BLOCKS_DIR) if os.path.isdir(os.path.join(BLOCKS_DIR, d))]
print(f"🧪 Phát hiện {len(patient_dirs)} bệnh nhân")

for p_dir in patient_dirs:
    ds = PatientBlockDataset(p_dir, seq_len=SEQ_LEN)
    if len(ds) > 0:
        all_datasets.append(ds)

if not all_datasets:
    raise ValueError("❌ Không tìm thấy dữ liệu blocks")

full_dataset = ConcatDataset(all_datasets)
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Mô hình
model = ConvNeXtLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Huấn luyện
for epoch in range(EPOCHS):
    model.train()
    all_preds, all_labels = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = pred.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")
