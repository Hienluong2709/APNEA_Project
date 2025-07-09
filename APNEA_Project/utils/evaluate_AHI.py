import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_ahi_from_y_blocks_dir(block_dir: str, block_duration_sec: int = 30):
    """
    Tính AHI từ các file y_*.npy trong thư mục block_dir.
    
    Args:
        block_dir: Đường dẫn tới thư mục chứa các file y_*.npy
        block_duration_sec: Thời lượng mỗi block (mặc định 30s)
    
    Returns:
        ahi: Apnea-Hypopnea Index
        apnea_count: Số block có apnea (nhãn = 1)
        total_blocks: Tổng số block
    """
    y_files = sorted([f for f in os.listdir(block_dir) if f.startswith('y_') and f.endswith('.npy')])

    apnea_count = 0
    total_blocks = 0

    for y_file in y_files:
        y_path = os.path.join(block_dir, y_file)
        y = np.load(y_path)

        # Trường hợp y là array (1,) hoặc scalar
        if isinstance(y, np.ndarray):
            y = y.item() if y.size == 1 else int(np.mean(y) > 0.5)

        apnea_count += int(y)
        total_blocks += 1

    total_hours = (total_blocks * block_duration_sec) / 3600
    ahi = apnea_count / total_hours if total_hours > 0 else 0

    return ahi, apnea_count, total_blocks

def calculate_ahi_from_predictions(y_true, y_pred, block_duration_sec: int = 30):
    """
    Tính AHI từ nhãn thực tế và nhãn dự đoán.
    
    Args:
        y_true: Mảng nhãn thực tế (0 hoặc 1)
        y_pred: Mảng nhãn dự đoán (0 hoặc 1) 
        block_duration_sec: Thời lượng mỗi block (mặc định 30s)
    
    Returns:
        true_ahi: AHI từ nhãn thực tế
        pred_ahi: AHI từ nhãn dự đoán
        metrics: Dictionary các chỉ số đánh giá (MAE, RMSE)
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
    metrics = {
        "mae": abs(true_ahi - pred_ahi),
        "rmse": np.sqrt((true_ahi - pred_ahi)**2)
    }
    
    return true_ahi, pred_ahi, metrics

def classify_osa_severity(ahi):
    """
    Phân loại mức độ nghiêm trọng của OSA dựa trên AHI
    
    Args:
        ahi: Apnea-Hypopnea Index
    
    Returns:
        severity: Mức độ nghiêm trọng (Normal, Mild, Moderate, Severe)
    """
    if ahi < 5:
        return "Normal"
    elif ahi < 15:
        return "Mild"
    elif ahi < 30:
        return "Moderate"
    else:
        return "Severe"

def evaluate_ahi_prediction_accuracy(true_ahis, pred_ahis):
    """
    Đánh giá độ chính xác của phân loại OSA dựa trên AHI
    
    Args:
        true_ahis: List các giá trị AHI thực tế
        pred_ahis: List các giá trị AHI dự đoán
    
    Returns:
        accuracy: Tỷ lệ phân loại đúng
        confusion: Confusion matrix của phân loại
    """
    true_classes = [classify_osa_severity(ahi) for ahi in true_ahis]
    pred_classes = [classify_osa_severity(ahi) for ahi in pred_ahis]
    
    # Tính độ chính xác
    correct = sum(1 for t, p in zip(true_classes, pred_classes) if t == p)
    accuracy = correct / len(true_classes)
    
    # Tạo confusion matrix
    classes = ["Normal", "Mild", "Moderate", "Severe"]
    confusion = np.zeros((4, 4), dtype=int)
    
    for t, p in zip(true_classes, pred_classes):
        t_idx = classes.index(t)
        p_idx = classes.index(p)
        confusion[t_idx, p_idx] += 1
    
    return accuracy, confusion, classes

def plot_severity_confusion_matrix(true_ahis, pred_ahis, save_path=None):
    """
    Vẽ confusion matrix của phân loại mức độ nghiêm trọng OSA
    
    Args:
        true_ahis: List các giá trị AHI thực tế
        pred_ahis: List các giá trị AHI dự đoán
        save_path: Đường dẫn để lưu biểu đồ
    """
    accuracy, confusion, classes = evaluate_ahi_prediction_accuracy(true_ahis, pred_ahis)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'OSA Severity Classification Confusion Matrix\nAccuracy: {accuracy:.2f}')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Hiển thị giá trị trong confusion matrix
    thresh = confusion.max() / 2
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, confusion[i, j],
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Mức độ thực tế')
    plt.xlabel('Mức độ dự đoán')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu confusion matrix tại {save_path}")
    else:
        plt.show()
