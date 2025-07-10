"""
Helper script để chạy train_transformer_improved.py từ thư mục train
"""

import os
import sys
import argparse

def main():
    # Lấy đường dẫn tới thư mục chứa script này
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Phân tích tham số
    parser = argparse.ArgumentParser(description='Chạy huấn luyện mô hình ConvNeXtTransformerLite với cải tiến')
    parser.add_argument('--data_dir', type=str, default='../data/blocks', help='Đường dẫn đến thư mục chứa dữ liệu')
    parser.add_argument('--batch_size', type=int, default=64, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=30, help='Số epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use_amp', action='store_true', help='Sử dụng Automatic Mixed Precision')
    parser.add_argument('--use_mixup', action='store_true', help='Sử dụng kỹ thuật MixUp')
    parser.add_argument('--use_swa', action='store_true', help='Sử dụng Stochastic Weight Averaging')
    parser.add_argument('--use_augment', action='store_true', help='Sử dụng data augmentation')
    parser.add_argument('--balance_classes', action='store_true', help='Cân bằng các lớp')
    
    args = parser.parse_args()
    
    # Đảm bảo data_dir là đường dẫn tuyệt đối
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(project_root, args.data_dir))
    
    # Tạo lệnh chạy
    cmd_args = [
        sys.executable,
        os.path.join(current_dir, "train_transformer_improved.py"),
        f"--data_dir={args.data_dir}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}"
    ]
    
    # Thêm các flag
    if args.use_amp:
        cmd_args.append("--use_amp")
    if args.use_mixup:
        cmd_args.append("--use_mixup")
    if args.use_swa:
        cmd_args.append("--use_swa")
    if args.use_augment:
        cmd_args.append("--use_augment")
    if args.balance_classes:
        cmd_args.append("--balance_classes")
    
    # Chạy lệnh
    import subprocess
    process = subprocess.run(cmd_args)
    
    # Trả về mã exit
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
