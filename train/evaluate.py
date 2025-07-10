"""
Script đánh giá mô hình ConvNeXtTransformerLite
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các module
from models.convnext_transformer_lite import ConvNeXtTransformerLite
from dataset.lazy_apnea_dataset import LazyApneaSingleDataset as LazyApneaDataset
try:
    from utils.visualization import plot_confusion_matrix, plot_ahi_comparison
except ImportError:
    print("⚠️ Các module trong utils không tìm thấy, đang định nghĩa lại các hàm cần thiết...")
    
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

def evaluate_model(model, test_loader, device='cuda'):
    """
    Đánh giá mô hình trên tập test
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Evaluating")):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds = pred.argmax(1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    
    # Tính các metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Tính AHI và so sánh
    true_ahi, pred_ahi, ahi_metrics = calculate_ahi_from_predictions(all_labels, all_preds)
    
    # Tính phân loại OSA dựa trên AHI
    true_severity = classify_osa_severity(true_ahi)
    pred_severity = classify_osa_severity(pred_ahi)
    
    # Phân bố nhãn và dự đoán
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    
    # In kết quả
    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AHI thực tế: {true_ahi:.2f}, Mức độ: {true_severity}")
    print(f"AHI dự đoán: {pred_ahi:.2f}, Mức độ: {pred_severity}")
    print(f"MAE AHI: {ahi_metrics['mae']:.2f}")
    print(f"RMSE AHI: {ahi_metrics['rmse']:.2f}")
    print(f"Phân bố nhãn thực tế: {dict(zip(unique_labels, label_counts))}")
    print(f"Phân bố nhãn dự đoán: {dict(zip(unique_preds, pred_counts))}")
    
    # Vẽ confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    
    # Tạo thư mục kết quả
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Vẽ biểu đồ
    plot_confusion_matrix(
        all_labels, all_preds, 
        classes=['Không apnea', 'Apnea'], 
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # Lưu kết quả dưới dạng CSV
    results = {
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'True AHI', 'Predicted AHI', 'MAE AHI', 'RMSE AHI'],
        'Value': [accuracy, f1, precision, recall, true_ahi, pred_ahi, ahi_metrics['mae'], ahi_metrics['rmse']]
    }
    pd.DataFrame(results).to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_ahi': true_ahi,
        'pred_ahi': pred_ahi,
        'mae_ahi': ahi_metrics['mae'],
        'rmse_ahi': ahi_metrics['rmse'],
        'confusion_matrix': cm
    }

def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình ConvNeXtTransformerLite')
    parser.add_argument('--data_dir', type=str, default='../data/blocks', help='Đường dẫn đến thư mục chứa dữ liệu test')
    parser.add_argument('--model_path', type=str, default=None, help='Đường dẫn đến file mô hình đã huấn luyện')
    parser.add_argument('--batch_size', type=int, default=64, help='Kích thước batch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Thiết bị đánh giá')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Khởi tạo random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Chuyển đường dẫn tương đối thành tuyệt đối
    data_dir = os.path.abspath(args.data_dir)
    
    # Tạo dataset và dataloader
    test_dataset = LazyApneaDataset(data_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"📊 Test dataset: {len(test_dataset)} mẫu")
    
    # Khởi tạo mô hình
    model = ConvNeXtTransformerLite(
        num_classes=2, 
        embed_dim=128, 
        num_heads=8, 
        num_transformer_layers=4, 
        dropout=0.3
    ).to(args.device)
    
    # Tìm mô hình đã huấn luyện
    if args.model_path is None:
        # Tìm trong thư mục checkpoints
        checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
        
        # Ưu tiên: SWA > Best_F1 > Best
        model_paths = [
            os.path.join(checkpoints_dir, 'ConvNeXtTransformerLite_swa.pth'),
            os.path.join(checkpoints_dir, 'ConvNeXtTransformerLite_best_f1.pth'),
            os.path.join(checkpoints_dir, 'ConvNeXtTransformerLite_best.pth')
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                args.model_path = path
                break
        
        if args.model_path is None:
            print("❌ Không tìm thấy mô hình đã huấn luyện!")
            return
    
    # Load mô hình
    print(f"📂 Đang load mô hình từ {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    # Đánh giá mô hình
    print(f"🧪 Đang đánh giá mô hình trên {args.device}...")
    metrics = evaluate_model(model, test_loader, device=args.device)
    
    print("✅ Đánh giá hoàn tất!")

if __name__ == "__main__":
    main()
