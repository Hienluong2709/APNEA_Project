import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Thêm thư mục gốc vào sys.path để import models và dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from torchvision.models.convnext import convnext_tiny
except ModuleNotFoundError:
    raise ImportError("❌ Bạn chưa cài torchvision. Vui lòng chạy: pip install torchvision")

from models.convnext_lstm import ConvNeXtLSTM
from dataset.lazy_apnea_dataset import LazyApneaDataset
from sklearn.metrics import accuracy_score, f1_score

# Config
BLOCK_DIR = "data/blocks"
BATCH_SIZE = 8
EPOCHS = 10

# Load dataset từ nhiều block theo cách tiết kiệm RAM
train_set = LazyApneaDataset(block_dir=BLOCK_DIR)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)

# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Train loop
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
