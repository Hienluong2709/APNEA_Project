# Phát hiện Apnea với ConvNeXt-LSTM và ConvNeXt-Transformer

Dự án này sử dụng các mô hình học sâu ConvNeXt kết hợp với LSTM và Transformer để phát hiện tiếng ngáy và gián đoạn hô hấp khi ngủ (Sleep Apnea) từ tín hiệu âm thanh.

## Cấu trúc thư mục

```
APNEA_Project/
├── preprocessing/
│   ├── build_dataset.py          # Tạo mel-spectrogram từ dữ liệu âm thanh gốc
│   └── melspec_utils.py          # Các hàm tiện ích cho xử lý mel-spectrogram
├── models/
│   ├── convnext_lite.py          # Backbone ConvNeXt nhẹ (~2M parameters)
│   ├── convnext_lstm_lite.py     # Mô hình ConvNeXt-LSTM nhẹ
│   └── convnext_transformer_lite.py  # Mô hình ConvNeXt-Transformer nhẹ
├── dataset/
│   └── lazy_apnea_dataset.py     # Dataset hiệu quả về bộ nhớ (lazy loading)
├── train/
│   ├── train_lite.py             # Script huấn luyện các mô hình nhẹ
│   └── evaluate.py               # Đánh giá mô hình trên tập test
├── utils/
│   └── metrics.py                # Các hàm tính toán metrics
├── run_pipeline.py               # Script chạy toàn bộ quy trình từ đầu đến cuối
└── check_model_size.py           # Kiểm tra kích thước các mô hình
```

## Cài đặt môi trường

```bash
# Tạo môi trường Python mới (tùy chọn)
conda create -n apnea python=3.8
conda activate apnea

# Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn tqdm librosa
```

## Quy trình chạy đầy đủ

Bạn có thể chạy toàn bộ quy trình (tiền xử lý dữ liệu, kiểm tra mô hình, huấn luyện) với lệnh:

```bash
python run_pipeline.py
```

Hoặc chạy từng bước riêng lẻ:

```bash
# Chỉ tiền xử lý dữ liệu
python run_pipeline.py --preprocess

# Chỉ kiểm tra kích thước mô hình
python run_pipeline.py --check_model

# Chỉ huấn luyện mô hình
python run_pipeline.py --train

# Huấn luyện mô hình cụ thể
python run_pipeline.py --train --model lstm
python run_pipeline.py --train --model transformer
```

## Kiểm tra kích thước mô hình

Để kiểm tra kích thước (số lượng tham số) của các mô hình:

```bash
python check_model_size.py
```

## Tiền xử lý dữ liệu

Tiền xử lý dữ liệu để tạo mel-spectrogram:

```bash
python preprocessing/build_dataset.py
```

Dữ liệu sau khi xử lý sẽ được lưu trong thư mục `data/blocks/` theo từng bệnh nhân.

## Huấn luyện mô hình

Để huấn luyện mô hình:

```bash
# Huấn luyện ConvNeXt-LSTM
python train/train_lite.py --model lstm --batch_size 32 --epochs 20

# Huấn luyện ConvNeXt-Transformer
python train/train_lite.py --model transformer --batch_size 32 --epochs 20
```

## Các mô hình đã cài đặt

1. **ConvNeXtBackboneLite**: Backbone nhẹ (~2M tham số) dựa trên kiến trúc ConvNeXt.
2. **ConvNeXtLSTMLite**: Kết hợp ConvNeXtBackboneLite với LSTM.
3. **ConvNeXtTransformerLite**: Kết hợp ConvNeXtBackboneLite với Transformer.

Các mô hình đều sử dụng backbone chung và có khoảng 2-2.5M tham số, phù hợp với yêu cầu về kích thước nhỏ gọn.
