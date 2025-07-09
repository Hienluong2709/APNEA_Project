import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
from .evaluate_AHI import calculate_ahi_from_y_blocks_dir
import seaborn as sns

def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, metric_name='F1', save_path=None):
    """
    Vẽ biểu đồ lịch sử huấn luyện (loss và metrics)
    
    Args:
        train_losses: list của loss trên tập train
        val_losses: list của loss trên tập validation
        train_metrics: list của metric trên tập train
        val_metrics: list của metric trên tập validation
        metric_name: tên của metric (mặc định: F1)
        save_path: đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    plt.figure(figsize=(15, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss trong quá trình huấn luyện')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.title(f'{metric_name} trong quá trình huấn luyện')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ huấn luyện tại {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Apnea'], save_path=None):
    """
    Vẽ confusion matrix
    
    Args:
        y_true: nhãn thực tế
        y_pred: nhãn dự đoán
        classes: tên các lớp
        save_path: đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    
    # Tính và hiển thị các metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}',
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu confusion matrix tại {save_path}")
    else:
        plt.show()

def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    Vẽ đường cong ROC
    
    Args:
        y_true: nhãn thực tế
        y_scores: xác suất dự đoán của lớp dương
        save_path: đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu ROC curve tại {save_path}")
    else:
        plt.show()

def plot_ahi_comparison(y_true_ahi, y_pred_ahi, save_path=None):
    """
    Vẽ biểu đồ so sánh AHI thực tế và dự đoán (như trong hình bạn đã đính kèm)
    
    Args:
        y_true_ahi: list các giá trị AHI thực tế
        y_pred_ahi: list các giá trị AHI dự đoán
        save_path: đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Tính PCC (Pearson Correlation Coefficient)
    pcc = np.corrcoef(y_true_ahi, y_pred_ahi)[0, 1]
    
    # Scatter plot với đường xu hướng
    ax1.scatter(y_true_ahi, y_pred_ahi, alpha=0.7, edgecolors='k', s=100, c='teal')
    max_val = max(max(y_true_ahi), max(y_pred_ahi))
    ax1.plot([0, max_val], [0, max_val], 'k-')
    ax1.set_xlabel('AHI-PSG (events/h)')
    ax1.set_ylabel('AHI-estimation (events/h)')
    ax1.text(max_val*0.1, max_val*0.8, f'PCC = {pcc:.3f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_val*1.05)
    ax1.set_ylim(0, max_val*1.05)
    
    # Bland-Altman plot
    differences = np.array(y_pred_ahi) - np.array(y_true_ahi)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    mean_values = [(a + b) / 2 for a, b in zip(y_true_ahi, y_pred_ahi)]
    
    ax2.scatter(mean_values, differences, alpha=0.7, edgecolors='k', s=100)
    ax2.axhline(mean_diff, color='purple', linestyle='-.')
    ax2.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--')
    ax2.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--')
    
    ax2.text(max(mean_values)*0.95, mean_diff, f'Mean\n{mean_diff:.3f}', fontsize=12, 
             verticalalignment='center', horizontalalignment='right')
    ax2.text(max(mean_values)*0.95, mean_diff + 1.96*std_diff, f'1.96 SD\n{(mean_diff + 1.96*std_diff):.3f}', 
             fontsize=12, verticalalignment='center', horizontalalignment='right')
    ax2.text(max(mean_values)*0.95, mean_diff - 1.96*std_diff, f'-1.96 SD\n{(mean_diff - 1.96*std_diff):.3f}', 
             fontsize=12, verticalalignment='center', horizontalalignment='right')
    
    ax2.set_xlabel('Mean value of AHI_pred and AHI_PSG')
    ax2.set_ylabel('Difference')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ AHI comparison tại {save_path}")
    else:
        plt.show()

def evaluate_ahi_predictions(true_dirs, pred_dirs, patient_ids=None, save_path=None):
    """
    Đánh giá và vẽ biểu đồ so sánh AHI từ nhiều bệnh nhân
    
    Args:
        true_dirs: list các thư mục chứa nhãn thực tế (y_*.npy)
        pred_dirs: list các thư mục chứa nhãn dự đoán (y_pred_*.npy)
        patient_ids: list IDs của bệnh nhân tương ứng
        save_path: đường dẫn để lưu biểu đồ, nếu None thì hiển thị
    """
    if patient_ids is None:
        patient_ids = [os.path.basename(d) for d in true_dirs]
    
    true_ahis = []
    pred_ahis = []
    
    for true_dir, pred_dir in zip(true_dirs, pred_dirs):
        true_ahi, _, _ = calculate_ahi_from_y_blocks_dir(true_dir)
        pred_ahi, _, _ = calculate_ahi_from_y_blocks_dir(pred_dir)
        
        true_ahis.append(true_ahi)
        pred_ahis.append(pred_ahi)
    
    # Vẽ biểu đồ so sánh
    plot_ahi_comparison(true_ahis, pred_ahis, save_path)
    
    # Tạo dataframe kết quả
    results = pd.DataFrame({
        'Patient ID': patient_ids,
        'True AHI': true_ahis,
        'Predicted AHI': pred_ahis,
        'Difference': np.array(pred_ahis) - np.array(true_ahis)
    })
    
    # Tính các chỉ số đánh giá
    mae = np.mean(np.abs(results['Difference']))
    rmse = np.sqrt(np.mean(np.square(results['Difference'])))
    pcc = np.corrcoef(true_ahis, pred_ahis)[0, 1]
    
    print("\n===== ĐÁNH GIÁ DỰ ĐOÁN AHI =====")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}")
    
    return results, mae, rmse, pcc
