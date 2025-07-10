"""
Script chính để chạy pipeline huấn luyện và đánh giá mô hình ConvNeXtTransformerLite
với các cải tiến nâng cao
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    # Thêm đường dẫn gốc vào sys.path
    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root))
    
    # Đảm bảo thư mục hiện tại là thư mục gốc của dự án
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='Pipeline huấn luyện và đánh giá mô hình ConvNeXtTransformerLite')
    parser.add_argument('--data_dir', type=str, default='data/blocks', help='Đường dẫn đến thư mục chứa dữ liệu')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='both', 
                        help='Chế độ: train, evaluate hoặc cả hai')
    parser.add_argument('--batch_size', type=int, default=64, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=30, help='Số epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use_amp', action='store_true', help='Sử dụng Automatic Mixed Precision')
    parser.add_argument('--use_mixup', action='store_true', help='Sử dụng kỹ thuật MixUp')
    parser.add_argument('--use_swa', action='store_true', help='Sử dụng Stochastic Weight Averaging')
    parser.add_argument('--use_augment', action='store_true', help='Sử dụng data augmentation')
    parser.add_argument('--balance_classes', action='store_true', help='Cân bằng các lớp')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Đường dẫn tuyệt đối
    data_dir = os.path.abspath(os.path.join(project_root, args.data_dir))
    
    # Xây dựng lệnh huấn luyện
    train_cmd = [
        "python", "-m", "train.train_transformer_improved",
        f"--data_dir={data_dir}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--seed={args.seed}"
    ]
    
    # Thêm các flag
    if args.use_amp:
        train_cmd.append("--use_amp")
    if args.use_mixup:
        train_cmd.append("--use_mixup")
    if args.use_swa:
        train_cmd.append("--use_swa")
    if args.use_augment:
        train_cmd.append("--use_augment")
    if args.balance_classes:
        train_cmd.append("--balance_classes")
    
    # Xây dựng lệnh đánh giá
    eval_cmd = [
        "python", "-m", "train.evaluate",
        f"--data_dir={data_dir}",
        f"--batch_size={args.batch_size}",
        f"--seed={args.seed}"
    ]
    
    # Chạy các lệnh
    import subprocess
    
    if args.mode in ['train', 'both']:
        print("🚀 Bắt đầu huấn luyện mô hình ConvNeXtTransformerLite...")
        train_process = subprocess.run(" ".join(train_cmd), shell=True)
        
        if train_process.returncode != 0:
            print("❌ Huấn luyện thất bại!")
            return
    
    if args.mode in ['evaluate', 'both']:
        print("📊 Bắt đầu đánh giá mô hình...")
        eval_process = subprocess.run(" ".join(eval_cmd), shell=True)
        
        if eval_process.returncode != 0:
            print("❌ Đánh giá thất bại!")
            return
    
    print("✅ Pipeline hoàn tất!")

if __name__ == "__main__":
    main()
