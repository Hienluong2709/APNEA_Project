# Phân tích và Biểu đồ cho Mô hình Phát hiện Sleep Apnea

Hệ thống này cung cấp các công cụ để phân tích và trực quan hóa kết quả của các mô hình phát hiện Sleep Apnea, bao gồm các biểu đồ đánh giá hiệu suất và so sánh dự đoán AHI (Apnea-Hypopnea Index).

## Các biểu đồ được hỗ trợ

1. **Confusion Matrix**: Biểu diễn ma trận nhầm lẫn của mô hình trong phân loại Apnea
2. **ROC Curve**: Đường cong ROC và chỉ số AUC
3. **AHI Comparison**: So sánh AHI dự đoán với AHI thực tế (từ PSG), bao gồm:
   - Biểu đồ tương quan với hệ số tương quan Pearson (PCC)
   - Biểu đồ Bland-Altman đánh giá sự khác biệt
4. **OSA Severity Confusion Matrix**: Ma trận nhầm lẫn trong phân loại mức độ OSA (Normal, Mild, Moderate, Severe)
5. **Model Comparison**: So sánh các chỉ số hiệu suất giữa các mô hình khác nhau

## Cách sử dụng

### Đánh giá mô hình với biểu đồ

Sử dụng script `evaluate_models.py` để đánh giá và tạo biểu đồ:

```bash
# Đánh giá cả hai mô hình LSTM và Transformer
python evaluate_models.py --data_dir data/blocks --checkpoint_dir checkpoints --output_dir evaluation_results

# Chỉ đánh giá mô hình LSTM
python evaluate_models.py --lstm --data_dir data/blocks

# Chỉ đánh giá mô hình Transformer
python evaluate_models.py --transformer --data_dir data/blocks
```

### Sử dụng trong code

```python
from utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve, plot_ahi_comparison
from utils.evaluate_AHI import plot_severity_confusion_matrix, calculate_ahi_from_predictions

# Vẽ biểu đồ lịch sử huấn luyện
plot_training_history(
    train_losses, val_losses,
    train_f1_scores, val_f1_scores,
    metric_name='F1',
    save_path='training_history.png'
)

# Vẽ confusion matrix
plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')

# Vẽ ROC curve
plot_roc_curve(y_true, y_scores, save_path='roc_curve.png')

# Vẽ so sánh AHI
plot_ahi_comparison(true_ahis, pred_ahis, save_path='ahi_comparison.png')

# Vẽ ma trận nhầm lẫn mức độ OSA
plot_severity_confusion_matrix(true_ahis, pred_ahis, save_path='severity_cm.png')
```

## Chỉ số đánh giá

Script đánh giá sẽ tính toán các chỉ số sau:

### Đánh giá phân loại
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Đánh giá AHI
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Pearson Correlation Coefficient (PCC)

### Đánh giá mức độ OSA
- Tỷ lệ phân loại đúng mức độ OSA (Normal, Mild, Moderate, Severe)
- Ma trận nhầm lẫn của các mức độ OSA

## Cập nhật train_lite.py để lưu lịch sử huấn luyện

Để lưu lại lịch sử huấn luyện và sử dụng biểu đồ trực quan, bạn có thể cập nhật file `train_lite.py` để lưu lại các giá trị loss và metrics trong quá trình huấn luyện:

```python
# Trong hàm train_model của file train_lite.py, thêm:
train_losses, val_losses = [], []
train_f1s, val_f1s = [], []

for epoch in range(epochs):
    # ...mã huấn luyện hiện tại...
    
    # Lưu lại loss và metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

# Sau khi huấn luyện, vẽ biểu đồ
from utils.visualization import plot_training_history
plot_training_history(
    train_losses, val_losses, 
    train_f1s, val_f1s,
    metric_name='F1',
    save_path=f'checkpoints/{model.__class__.__name__}_history.png'
)
```

## Cấu trúc thư mục kết quả

Sau khi chạy script đánh giá, bạn sẽ có cấu trúc thư mục như sau:

```
evaluation_results/
├── lstm/
│   ├── ConvNeXtLSTMLite_confusion_matrix.png
│   ├── ConvNeXtLSTMLite_roc_curve.png
│   ├── ConvNeXtLSTMLite_ahi_comparison.png
│   ├── ConvNeXtLSTMLite_osa_severity_cm.png
│   └── ConvNeXtLSTMLite_metrics.json
├── transformer/
│   ├── ConvNeXtTransformerLite_confusion_matrix.png
│   ├── ConvNeXtTransformerLite_roc_curve.png
│   ├── ConvNeXtTransformerLite_ahi_comparison.png
│   ├── ConvNeXtTransformerLite_osa_severity_cm.png
│   └── ConvNeXtTransformerLite_metrics.json
└── model_comparison.png
```
