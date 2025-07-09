"""
Script đánh giá và so sánh kết quả của các mô hình đã huấn luyện,
kèm theo các biểu đồ trực quan để phân tích hiệu suất.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các module
from utils.visualization import (plot_training_history, plot_confusion_matrix, 
                              plot_roc_curve, plot_ahi_comparison, evaluate_ahi_predictions)
from utils.evaluate_AHI import calculate_ahi_from_y_blocks_dir, plot_severity_confusion_matrix
from models.convnext_lstm_lite import ConvNeXtLSTMLite
from models.convnext_transformer_lite import ConvNeXtTransformerLite
from dataset.lazy_apnea_dataset import LazyApneaDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

def load_model(model_type, checkpoint_path, device):
    """
    Tải mô hình từ checkpoint
    """
    if model_type.lower() == 'lstm':
        model = ConvNeXtLSTMLite(num_classes=2).to(device)
    elif model_type.lower() == 'transformer':
        model = ConvNeXtTransformerLite(num_classes=2).to(device)
    else:
        raise ValueError(f"Loại mô hình {model_type} không được hỗ trợ")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"✅ Đã tải mô hình {model_type} từ {checkpoint_path}")
    return model

def predict_dataset(model, data_loader, device):
    """
    Dự đoán trên toàn bộ dataset và trả về nhãn dự đoán và xác suất
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def evaluate_model(model, data_loader, device, output_dir, model_name):
    """
    Đánh giá mô hình và lưu các biểu đồ kết quả
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Đang đánh giá mô hình {model_name}...")
    y_true, y_pred, y_probs = predict_dataset(model, data_loader, device)
    
    # Tính các metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Lưu metrics vào file
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    }
    
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"📊 Kết quả đánh giá mô hình {model_name}:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    ROC AUC: {roc_auc:.4f}")
    
    # Vẽ confusion matrix
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # Vẽ ROC curve
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(y_true, y_probs, save_path=roc_path)
    
    return metrics

def evaluate_all_patients(model, data_dir, output_dir, model_name, device):
    """
    Đánh giá mô hình trên từng bệnh nhân riêng lẻ và tính AHI
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả các thư mục bệnh nhân
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    true_ahis = []
    pred_ahis = []
    patient_ids = []
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        try:
            # Tạo dataset cho bệnh nhân
            dataset = LazyApneaDataset(patient_dir)
            if len(dataset) == 0:
                continue
                
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Dự đoán
            y_true, y_pred, _ = predict_dataset(model, dataloader, device)
            
            # Tính AHI
            true_ahi = sum(y_true) / (len(y_true) * 30 / 3600)
            pred_ahi = sum(y_pred) / (len(y_pred) * 30 / 3600)
            
            true_ahis.append(true_ahi)
            pred_ahis.append(pred_ahi)
            patient_ids.append(patient_id)
            
            print(f"✅ Bệnh nhân {patient_id}: True AHI = {true_ahi:.2f}, Pred AHI = {pred_ahi:.2f}")
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý bệnh nhân {patient_id}: {e}")
    
    # Vẽ biểu đồ so sánh AHI
    if true_ahis and pred_ahis:
        ahi_path = os.path.join(output_dir, f"{model_name}_ahi_comparison.png")
        plot_ahi_comparison(true_ahis, pred_ahis, save_path=ahi_path)
        
        # Vẽ confusion matrix mức độ OSA
        severity_cm_path = os.path.join(output_dir, f"{model_name}_osa_severity_cm.png")
        plot_severity_confusion_matrix(true_ahis, pred_ahis, save_path=severity_cm_path)
    
    return true_ahis, pred_ahis, patient_ids

def compare_models(models_metrics, output_dir):
    """
    So sánh các mô hình bằng biểu đồ
    """
    models = list(models_metrics.keys())
    
    # So sánh các metrics
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    values = {metric: [models_metrics[model][metric] for model in models] for metric in metrics}
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, values[metric], width, label=metric.capitalize())
    
    plt.xlabel('Mô hình')
    plt.ylabel('Score')
    plt.title('So sánh hiệu suất các mô hình')
    plt.xticks(x + width*2, models)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ so sánh tại {os.path.join(output_dir, 'model_comparison.png')}")

def main(args):
    # Thiết lập đường dẫn
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Thiết lập device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"🖥️ Sử dụng thiết bị: {device}")
    
    # Các mô hình cần đánh giá
    models_to_evaluate = []
    
    if args.lstm:
        lstm_path = os.path.join(checkpoint_dir, "ConvNeXtLSTMLite_best.pth")
        if os.path.exists(lstm_path):
            models_to_evaluate.append(("lstm", lstm_path))
    
    if args.transformer:
        transformer_path = os.path.join(checkpoint_dir, "ConvNeXtTransformerLite_best.pth")
        if os.path.exists(transformer_path):
            models_to_evaluate.append(("transformer", transformer_path))
    
    # Đánh giá các mô hình
    all_metrics = {}
    
    for model_type, checkpoint_path in models_to_evaluate:
        model = load_model(model_type, checkpoint_path, device)
        model_output_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Đánh giá mô hình trên toàn bộ tập dữ liệu
        print(f"\n===== ĐÁNH GIÁ MÔ HÌNH {model_type.upper()} =====")
        
        # Đánh giá AHI trên từng bệnh nhân
        print(f"\n🩺 Đánh giá AHI trên từng bệnh nhân...")
        true_ahis, pred_ahis, patient_ids = evaluate_all_patients(
            model, data_dir, model_output_dir, model_type, device
        )
        
        if true_ahis and pred_ahis:
            # Tính các chỉ số đánh giá AHI
            mae = np.mean(np.abs(np.array(pred_ahis) - np.array(true_ahis)))
            rmse = np.sqrt(np.mean(np.square(np.array(pred_ahis) - np.array(true_ahis))))
            corr = np.corrcoef(true_ahis, pred_ahis)[0, 1]
            
            print(f"📊 Đánh giá AHI cho {model_type}:")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Correlation: {corr:.4f}")
            
            # Thêm vào metrics
            metrics = {
                "model": model_type,
                "mae": float(mae),
                "rmse": float(rmse),
                "correlation": float(corr)
            }
            
            all_metrics[model_type] = metrics
    
    # So sánh các mô hình
    if len(all_metrics) > 1:
        compare_models(all_metrics, output_dir)
    
    print(f"\n✅ Đánh giá hoàn tất. Kết quả được lưu tại {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh các mô hình phát hiện Apnea")
    parser.add_argument('--data_dir', type=str, default='data/blocks',
                        help='Thư mục chứa dữ liệu để đánh giá')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Thư mục chứa các model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Thư mục để lưu kết quả đánh giá')
    parser.add_argument('--lstm', action='store_true', help='Đánh giá mô hình LSTM')
    parser.add_argument('--transformer', action='store_true', help='Đánh giá mô hình Transformer')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Thiết bị để chạy đánh giá (cuda hoặc cpu)')
    
    args = parser.parse_args()
    
    if not (args.lstm or args.transformer):
        # Mặc định đánh giá cả hai mô hình nếu không chỉ định
        args.lstm = True
        args.transformer = True
    
    main(args)
