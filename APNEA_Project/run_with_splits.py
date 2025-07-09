"""
Script thực thi pipeline huấn luyện và đánh giá với các phương pháp
chia dữ liệu khác nhau để so sánh hiệu suất
"""

import os
import sys
import argparse
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

def run_training(model, split_type, train_ratio=0.8, epochs=20, batch_size=32, seed=42):
    """
    Chạy script huấn luyện với các thông số cụ thể
    """
    print(f"\n{'='*80}")
    print(f"🚀 Bắt đầu huấn luyện mô hình {model} với phương pháp chia dữ liệu {split_type}")
    print(f"{'='*80}")
    
    # Tạo lệnh chạy
    cmd = [
        "python", "train/train_with_splits.py",
        "--model", model,
        "--split_type", split_type,
        "--train_ratio", str(train_ratio),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--seed", str(seed)
    ]
    
    # Chạy lệnh
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy huấn luyện: {e}")
        return False

def compare_results(models=['lstm', 'transformer'], split_types=['random', 'dependent', 'independent']):
    """
    So sánh kết quả huấn luyện giữa các phương pháp chia dữ liệu
    """
    # Lưu kết quả F1 để so sánh
    results = {}
    
    # Duyệt qua các mô hình và phương pháp chia dữ liệu
    for model in models:
        results[model] = {}
        
        for split_type in split_types:
            # Đường dẫn đến file kết quả
            result_dir = os.path.join('results', f'{split_type}_split')
            history_file = os.path.join('results', f'{"ConvNeXtLSTMLite" if model == "lstm" else "ConvNeXtTransformerLite"}_history.npz')
            
            # Nếu không tìm thấy file kết quả, bỏ qua
            if not os.path.exists(history_file):
                print(f"⚠️ Không tìm thấy kết quả cho {model} với phương pháp {split_type}")
                continue
            
            # Đọc dữ liệu lịch sử huấn luyện
            history = np.load(history_file)
            val_f1s = history['val_f1s']
            best_f1 = val_f1s.max()
            
            # Lưu kết quả
            results[model][split_type] = best_f1
    
    # Hiển thị kết quả
    print("\n" + "="*80)
    print("📊 KẾT QUẢ SO SÁNH CÁC PHƯƠNG PHÁP CHIA DỮ LIỆU")
    print("="*80)
    
    # Tạo header
    header = ["Mô hình"] + split_types
    print(f"{header[0]:<15}", end="")
    for split in header[1:]:
        print(f"{split:<15}", end="")
    print()
    
    # In dòng ngăn cách
    print("-"*15*len(header))
    
    # In dữ liệu
    for model in models:
        print(f"{model:<15}", end="")
        for split in split_types:
            if split in results[model]:
                print(f"{results[model][split]:.4f}       ", end="")
            else:
                print(f"N/A            ", end="")
        print()
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.25
    index = np.arange(len(split_types))
    
    for i, model in enumerate(models):
        f1_values = [results[model].get(split, 0) for split in split_types]
        plt.bar(index + i*bar_width, f1_values, bar_width, label=model)
    
    plt.xlabel('Phương pháp chia dữ liệu')
    plt.ylabel('F1 Score')
    plt.title('So sánh F1 Score giữa các phương pháp chia dữ liệu')
    plt.xticks(index + bar_width/2, split_types)
    plt.legend()
    
    # Lưu biểu đồ
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/split_methods_comparison.png')
    plt.close()
    
    print(f"\n✅ Đã lưu biểu đồ so sánh tại results/comparison/split_methods_comparison.png")
    
    # Lưu kết quả dưới dạng JSON
    with open('results/comparison/split_methods_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Đã lưu kết quả so sánh dưới dạng JSON tại results/comparison/split_methods_comparison.json")

def main(args):
    """
    Hàm chính của script
    """
    # Kiểm tra các thư mục cần thiết
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints', exist_ok=True)
    
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    
    # Chạy huấn luyện cho từng phương pháp chia dữ liệu
    for split_type in args.split_types:
        # Chạy với LSTM
        if 'lstm' in args.models:
            run_training(
                model='lstm',
                split_type=split_type,
                train_ratio=args.train_ratio,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed
            )
        
        # Chạy với Transformer
        if 'transformer' in args.models:
            run_training(
                model='transformer',
                split_type=split_type,
                train_ratio=args.train_ratio,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed
            )
    
    # So sánh kết quả
    if args.compare:
        compare_results(models=args.models, split_types=args.split_types)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline với nhiều phương pháp chia dữ liệu khác nhau")
    parser.add_argument('--models', nargs='+', default=['lstm', 'transformer'], 
                       choices=['lstm', 'transformer'],
                       help='Danh sách các mô hình cần chạy')
    parser.add_argument('--split_types', nargs='+', default=['random', 'dependent', 'independent'],
                       choices=['random', 'dependent', 'independent'],
                       help='Danh sách các phương pháp chia dữ liệu')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='Tỷ lệ dữ liệu dùng cho huấn luyện (mặc định: 0.8)')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Số epochs cho mỗi lần huấn luyện')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Kích thước batch')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Seed ngẫu nhiên')
    parser.add_argument('--compare', action='store_true', 
                       help='So sánh kết quả sau khi huấn luyện')
    
    args = parser.parse_args()
    main(args)
