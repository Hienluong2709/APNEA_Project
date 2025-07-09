"""
Script huấn luyện mô hình ConvNeXt-LSTM và ConvNeXt-Transformer
phiên bản nhẹ (~2-2.5M tham số) cho phát hiện Apnea
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các module
from models.convnext_lstm_lite import ConvNeXtLSTMLite
from models.convnext_transformer_lite import ConvNeXtTransformerLite
from dataset.lazy_apnea_dataset import LazyApneaDataset
from models.convnext_lite import count_parameters
try:
    from utils.visualization import plot_training_history, plot_confusion_matrix
except ImportError:
    print("⚠️ Module visualization không khả dụng. Sẽ không vẽ biểu đồ.")
    plot_training_history = None
    plot_confusion_matrix = None

def train_model(model, train_loader, val_loader, epochs=10, lr=3e-4, device='cuda'):
    """
    Huấn luyện và đánh giá mô hình
    """
    print(f"🚀 Bắt đầu huấn luyện mô hình {model.__class__.__name__} trên {device}")
    print(f"🔢 Số lượng tham số: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
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
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = pred.argmax(1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                
                val_loss += loss.item()
                preds = pred.argmax(1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        
        # Cập nhật learning rate
        scheduler.step(val_f1)
        
        # Lưu mô hình tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{model.__class__.__name__}_best.pth')
            
            # Chi tiết kết quả
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            cm = confusion_matrix(all_labels, all_preds)
            print(f"Confusion Matrix:\n{cm}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        
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
    
    # Vẽ biểu đồ lịch sử huấn luyện
    if plot_training_history is not None:
        os.makedirs('results', exist_ok=True)
        # Biểu đồ Loss và F1
        plot_training_history(
            train_losses, val_losses, 
            train_f1s, val_f1s,
            metric_name='F1', 
            save_path=f'results/{model.__class__.__name__}_training_history.png'
        )
        
        # Biểu đồ Loss và Accuracy
        plot_training_history(
            train_losses, val_losses, 
            train_accs, val_accs,
            metric_name='Accuracy', 
            save_path=f'results/{model.__class__.__name__}_accuracy_history.png'
        )
    
    # Lưu lịch sử huấn luyện dưới dạng NumPy array
    os.makedirs('results', exist_ok=True)
    np.savez(
        f'results/{model.__class__.__name__}_history.npz',
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        train_accs=np.array(train_accs),
        val_accs=np.array(val_accs),
        train_f1s=np.array(train_f1s),
        val_f1s=np.array(val_f1s)
    )
    
    return best_val_f1

def main(args):
    # Đường dẫn dữ liệu
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_dir, "data", "blocks")
    
    if not os.path.exists(data_dir):
        print(f"❌ Không tìm thấy thư mục blocks tại {data_dir}")
        print("⚠️ Vui lòng chạy build_dataset.py trước")
        return
    
    # Tổng hợp dữ liệu từ tất cả các bệnh nhân
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"📊 Đang tải dữ liệu từ {len(patient_dirs)} bệnh nhân")
    
    all_datasets = []
    for p_dir in patient_dirs:
        try:
            ds = LazyApneaDataset(p_dir)
            if len(ds) > 0:
                all_datasets.append(ds)
                print(f"✅ Đã tải {len(ds)} mẫu từ {os.path.basename(p_dir)}")
        except Exception as e:
            print(f"❌ Lỗi khi tải dữ liệu từ {p_dir}: {e}")
    
    if not all_datasets:
        print("❌ Không có dữ liệu để huấn luyện. Vui lòng chạy build_dataset.py trước.")
        return
    
    # Tạo dataset hoàn chỉnh
    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    print(f"📈 Tổng số mẫu: {len(full_dataset)}")
    
    # Chia tập train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Chọn thiết bị
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"🖥️ Sử dụng thiết bị: {device}")
    
    # Chọn và huấn luyện mô hình
    if args.model == 'lstm':
        model = ConvNeXtLSTMLite(num_classes=2).to(device)
    elif args.model == 'transformer':
        model = ConvNeXtTransformerLite(num_classes=2).to(device)
    else:
        print(f"❌ Mô hình {args.model} không được hỗ trợ")
        return
    
    # Huấn luyện
    best_val_f1 = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        lr=args.learning_rate,
        device=device
    )
    
    print(f"✅ Huấn luyện hoàn tất. Best F1: {best_val_f1:.4f}")
    
    # Đánh giá trên tập validation và vẽ confusion matrix
    if plot_confusion_matrix is not None:
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                outputs = model(x)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        # Vẽ confusion matrix
        os.makedirs('results', exist_ok=True)
        plot_confusion_matrix(
            all_labels, 
            all_preds,
            classes=['Normal', 'Apnea'], 
            save_path=f'results/{model.__class__.__name__}_confusion_matrix.png'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện Apnea")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'],
                        help='Loại mô hình để huấn luyện (lstm hoặc transformer)')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=20, help='Số epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Thiết bị để huấn luyện (cuda hoặc cpu)')
    parser.add_argument('--num_workers', type=int, default=0, help='Số worker cho DataLoader')
    
    args = parser.parse_args()
    main(args)
