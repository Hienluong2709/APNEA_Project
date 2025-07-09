# Phương pháp chia dữ liệu cho phát hiện Apnea

Tài liệu này mô tả ba phương pháp chia dữ liệu chính được sử dụng trong dự án phát hiện Apnea. Mỗi phương pháp có những ưu điểm và nhược điểm riêng, đặc biệt là trong bối cảnh dữ liệu y tế như phát hiện rối loạn hô hấp trong giấc ngủ (Sleep Apnea).

## 1. Random Split (Chia ngẫu nhiên)

Phương pháp này chia dữ liệu một cách ngẫu nhiên thành tập huấn luyện (train) và tập kiểm tra (validation) mà không quan tâm đến việc các mẫu thuộc về bệnh nhân nào.

### Đặc điểm:
- **Đơn giản và dễ thực hiện**
- **Phân bố dữ liệu đồng đều** giữa các tập nếu dữ liệu gốc cân bằng
- **Tốc độ nhanh** trong quá trình huấn luyện và đánh giá

### Hạn chế:
- **Không phản ánh thực tế lâm sàng**: Trong thực tế, mô hình sẽ được áp dụng cho các bệnh nhân mới
- **Có thể dẫn đến kết quả quá lạc quan** vì mô hình đã "thấy" dữ liệu của tất cả bệnh nhân trong quá trình huấn luyện
- **Không đánh giá được khả năng tổng quát hóa** cho bệnh nhân mới

### Sử dụng khi:
- Dữ liệu có kích thước nhỏ
- Muốn so sánh nhanh các kiến trúc mô hình
- Đang trong giai đoạn phát triển ban đầu

## 2. Dependent Subject Split (Chia dữ liệu theo bệnh nhân phụ thuộc)

Phương pháp này chia dữ liệu của từng bệnh nhân riêng lẻ thành tập huấn luyện và tập kiểm tra. Mỗi bệnh nhân sẽ có dữ liệu xuất hiện trong cả hai tập.

### Đặc điểm:
- **Cân bằng hơn giữa các bệnh nhân**: Mỗi bệnh nhân đều có dữ liệu trong cả train và validation
- **Giảm thiểu sự chênh lệch** do đặc điểm cá nhân của bệnh nhân
- **Đánh giá được khả năng dự đoán** các đoạn mới từ bệnh nhân đã biết

### Hạn chế:
- **Vẫn có rò rỉ thông tin** về đặc điểm của bệnh nhân giữa tập huấn luyện và kiểm tra
- **Không phản ánh đúng tình huống thực tế** khi gặp bệnh nhân hoàn toàn mới
- **Kết quả có thể vẫn lạc quan hơn thực tế**

### Sử dụng khi:
- Số lượng bệnh nhân ít, không đủ để chia thành các nhóm riêng biệt
- Muốn đánh giá khả năng dự đoán các đoạn mới của bệnh nhân đã biết
- Khi đặc điểm cá nhân của bệnh nhân ảnh hưởng nhiều đến kết quả

## 3. Independent Subject Split (Chia dữ liệu theo bệnh nhân độc lập)

Phương pháp này chia các bệnh nhân thành hai nhóm riêng biệt: nhóm huấn luyện và nhóm kiểm tra. Toàn bộ dữ liệu của một bệnh nhân sẽ chỉ xuất hiện trong một tập duy nhất.

### Đặc điểm:
- **Phản ánh chính xác tình huống thực tế lâm sàng**: Mô hình sẽ được đánh giá trên bệnh nhân hoàn toàn mới
- **Đánh giá chính xác hơn khả năng tổng quát hóa** của mô hình
- **Kết quả đáng tin cậy hơn** cho ứng dụng thực tế

### Hạn chế:
- **Yêu cầu nhiều dữ liệu hơn**, đặc biệt là nhiều bệnh nhân khác nhau
- **Kết quả có thể thay đổi nhiều** tùy thuộc vào cách chia nhóm
- **Thường cho kết quả thấp hơn** so với hai phương pháp trên

### Sử dụng khi:
- Có đủ dữ liệu từ nhiều bệnh nhân khác nhau
- Muốn đánh giá chính xác khả năng ứng dụng vào thực tế lâm sàng
- Đang trong giai đoạn đánh giá cuối cùng trước khi triển khai

## So sánh các phương pháp

| Tiêu chí | Random Split | Dependent Subject | Independent Subject |
|----------|-------------|-------------------|---------------------|
| Độ phức tạp | Thấp | Trung bình | Cao |
| Thời gian thực hiện | Nhanh | Trung bình | Chậm |
| Tính đại diện thực tế | Thấp | Trung bình | Cao |
| Khả năng tổng quát hóa | Kém | Trung bình | Tốt |
| Yêu cầu dữ liệu | Ít | Trung bình | Nhiều |
| Độ tin cậy kết quả | Thấp | Trung bình | Cao |

## Hướng dẫn sử dụng

Để sử dụng các phương pháp chia dữ liệu trong dự án, bạn có thể sử dụng script `train_with_splits.py` với tham số `--split_type` tương ứng:

```bash
# Huấn luyện với phương pháp chia ngẫu nhiên
python train/train_with_splits.py --model lstm --split_type random

# Huấn luyện với phương pháp chia theo bệnh nhân phụ thuộc
python train/train_with_splits.py --model lstm --split_type dependent

# Huấn luyện với phương pháp chia theo bệnh nhân độc lập
python train/train_with_splits.py --model lstm --split_type independent
```

Để chạy tất cả các phương pháp và so sánh kết quả, bạn có thể sử dụng script `run_with_splits.py`:

```bash
# Chạy tất cả các phương pháp với cả hai mô hình và so sánh kết quả
python run_with_splits.py --compare

# Chỉ chạy với một mô hình cụ thể
python run_with_splits.py --models lstm --compare

# Chỉ chạy một số phương pháp chia dữ liệu cụ thể
python run_with_splits.py --split_types random independent --compare
```

## Lời khuyên

1. **Bắt đầu với Random Split** trong giai đoạn phát triển ban đầu để nhanh chóng so sánh các kiến trúc mô hình.

2. **Sử dụng Dependent Subject Split** khi số lượng bệnh nhân ít, hoặc khi muốn đánh giá mô hình trên các đoạn mới của bệnh nhân đã biết.

3. **Luôn đánh giá cuối cùng bằng Independent Subject Split** để có kết quả đáng tin cậy nhất về khả năng ứng dụng vào thực tế lâm sàng.

4. **Thực hiện Cross-validation** với Independent Subject Split nếu có thể, để giảm thiểu ảnh hưởng của việc chia ngẫu nhiên bệnh nhân.

5. **Kết hợp nhiều phương pháp** để có cái nhìn toàn diện về hiệu suất của mô hình trong các tình huống khác nhau.
