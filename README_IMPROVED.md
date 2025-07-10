# Phát hiện Sleep Apnea với ConvNeXtTransformerLite

Dự án này sử dụng mô hình deep learning dựa trên kiến trúc ConvNeXt và Transformer để phát hiện tình trạng ngưng thở khi ngủ (Sleep Apnea) từ dữ liệu âm thanh.

## Cấu trúc dự án

```
APNEA_Project/
├── checkpoints/                  # Thư mục lưu trữ các mô hình đã huấn luyện
├── data/                         # Dữ liệu
│   └── blocks/                   # Các block mel-spectrogram đã tiền xử lý
├── dataset/                      # Module dataset
│   ├── __init__.py
│   └── lazy_apnea_dataset.py     # Các lớp dataset với lazy loading và augmentation
├── evaluation_results/           # Kết quả đánh giá
├── models/                       # Định nghĩa các mô hình
│   ├── __init__.py
│   ├── convnext_lite.py          # ConvNeXT backbone nhẹ với stochastic depth
│   └── convnext_transformer_lite.py  # Mô hình kết hợp ConvNeXT và Transformer
├── preprocessing/                # Module tiền xử lý
│   ├── build_dataset.py
│   └── melspec_utils.py
├── results/                      # Kết quả huấn luyện (biểu đồ, metrics)
├── train/                        # Module huấn luyện
│   ├── evaluate.py               # Script đánh giá mô hình
│   └── train_transformer_improved.py # Script huấn luyện cải tiến
├── utils/                        # Module tiện ích
│   ├── data_splitting.py
│   ├── evaluate_AHI.py
│   ├── metrics.py
│   └── visualization.py
├── run_improved_pipeline.py      # Script chạy pipeline cải tiến
└── requirements.txt              # Các thư viện cần thiết
```

## Các cải tiến

Dự án này đã được cải tiến với nhiều kỹ thuật tiên tiến để nâng cao hiệu suất và khả năng tổng quát hóa:

1. **Cải tiến kiến trúc mô hình**:
   - Tăng kích thước embedding và số lượng heads trong Transformer
   - Thêm Focal Attention để tập trung vào các đặc trưng quan trọng
   - Áp dụng Stochastic Depth (Dropout Path) cho backbone ConvNeXT
   - Thêm Pre-Normalization cho các lớp Transformer
   - Bổ sung BatchNorm và LayerNorm tại các vị trí chiến lược

2. **Kỹ thuật huấn luyện nâng cao**:
   - Class weighting để xử lý mất cân bằng dữ liệu
   - AdamW optimizer với weight decay tốt hơn
   - Gradient clipping để ổn định huấn luyện
   - Exponential Moving Average (EMA) cho trọng số mô hình
   - Lập lịch learning rate thông minh (OneCycleLR)
   - Early Stopping để tránh overfitting
   - Stochastic Weight Averaging (SWA) để cải thiện khả năng tổng quát hóa
   - Automatic Mixed Precision (AMP) để tăng tốc huấn luyện

3. **Cải tiến dataset**:
   - Data augmentation với time mask, frequency mask và gaussian noise
   - Class balancing bằng upsampling lớp thiểu số
   - Kỹ thuật MixUp cho augmentation nâng cao

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd APNEA_Project
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Sử dụng

### Huấn luyện mô hình

Để huấn luyện mô hình với các cài đặt mặc định:

```bash
python run_improved_pipeline.py --data_dir=data/blocks
```

Với các tùy chọn nâng cao:

```bash
python run_improved_pipeline.py --data_dir=data/blocks --batch_size=64 --epochs=30 \
    --use_amp --use_mixup --use_swa --use_augment --balance_classes
```

### Đánh giá mô hình

Để đánh giá mô hình riêng biệt:

```bash
python -m train.evaluate --data_dir=data/blocks
```

### Các tùy chọn

- `--data_dir`: Đường dẫn đến thư mục chứa dữ liệu
- `--batch_size`: Kích thước batch (mặc định: 64)
- `--epochs`: Số epochs huấn luyện (mặc định: 30)
- `--lr`: Learning rate (mặc định: 3e-4)
- `--use_amp`: Sử dụng Automatic Mixed Precision
- `--use_mixup`: Sử dụng kỹ thuật MixUp
- `--use_swa`: Sử dụng Stochastic Weight Averaging
- `--use_augment`: Sử dụng data augmentation
- `--balance_classes`: Cân bằng các lớp bằng upsampling
- `--seed`: Random seed (mặc định: 42)

## Kết quả

Các cải tiến đã giúp mô hình:
- Giảm đáng kể overfitting
- Tăng F1 score và accuracy trên tập validation
- Dự đoán AHI chính xác hơn
- Phân phối nhãn dự đoán cân bằng hơn

## Tài liệu tham khảo

- [ConvNeXT: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Stochastic Depth in Deep Networks](https://arxiv.org/abs/1603.09382)
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
