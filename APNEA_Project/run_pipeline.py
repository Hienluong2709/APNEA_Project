"""
Script chạy toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện
Sử dụng các mô hình lite (~2-2.5M tham số)
"""
import os
import sys
import subprocess
import argparse
import time

def run_command(cmd, desc=None):
    """Chạy lệnh và in kết quả"""
    if desc:
        print(f"\n{'='*80}\n{desc}\n{'='*80}")
    
    print(f"Đang chạy: {cmd}")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        shell=True, 
        text=True
    )
    
    # In kết quả theo thời gian thực
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    elapsed = time.time() - start_time
    print(f"\nĐã hoàn thành sau {elapsed:.2f} giây với mã trạng thái: {process.returncode}")
    
    if process.returncode != 0:
        print(f"⚠️ Cảnh báo: Lệnh kết thúc với mã lỗi {process.returncode}")
    
    return process.returncode

def main(args):
    # Thiết lập đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # 1. Xử lý dữ liệu
    if args.run_all or args.preprocess:
        preprocess_cmd = f"python {os.path.join(project_dir, 'preprocessing', 'build_dataset.py')}"
        if run_command(preprocess_cmd, "1. TIỀN XỬ LÝ DỮ LIỆU") != 0:
            print("❌ Tiền xử lý dữ liệu thất bại. Dừng quy trình.")
            return
    
    # 2. Kiểm tra kích thước mô hình
    if args.run_all or args.check_model:
        check_model_cmd = f"python {os.path.join(project_dir, 'check_model_size.py')}"
        if run_command(check_model_cmd, "2. KIỂM TRA KÍCH THƯỚC MÔ HÌNH") != 0:
            print("⚠️ Kiểm tra mô hình thất bại, nhưng quy trình vẫn tiếp tục.")
    
    # 3. Huấn luyện các mô hình
    if args.run_all or args.train:
        # Huấn luyện mô hình LSTM
        if args.model in ['all', 'lstm']:
            lstm_cmd = f"python {os.path.join(project_dir, 'train', 'train_lite.py')} --model lstm --batch_size {args.batch_size} --epochs {args.epochs} --device {args.device}"
            run_command(lstm_cmd, "3A. HUẤN LUYỆN MÔ HÌNH CONVNEXT-LSTM")
        
        # Huấn luyện mô hình Transformer
        if args.model in ['all', 'transformer']:
            transformer_cmd = f"python {os.path.join(project_dir, 'train', 'train_lite.py')} --model transformer --batch_size {args.batch_size} --epochs {args.epochs} --device {args.device}"
            run_command(transformer_cmd, "3B. HUẤN LUYỆN MÔ HÌNH CONVNEXT-TRANSFORMER")
    
    print(f"\n{'='*80}\nQUY TRÌNH ĐÃ HOÀN THÀNH\n{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ quy trình huấn luyện mô hình Apnea")
    
    # Lựa chọn quy trình
    parser.add_argument('--run_all', action='store_true', help='Chạy toàn bộ quy trình')
    parser.add_argument('--preprocess', action='store_true', help='Chỉ chạy tiền xử lý dữ liệu')
    parser.add_argument('--check_model', action='store_true', help='Chỉ kiểm tra kích thước mô hình')
    parser.add_argument('--train', action='store_true', help='Chỉ chạy huấn luyện mô hình')
    
    # Tham số huấn luyện
    parser.add_argument('--model', type=str, default='all', choices=['all', 'lstm', 'transformer'],
                        help='Mô hình để huấn luyện (all, lstm hoặc transformer)')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=20, help='Số epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Thiết bị để huấn luyện (cuda hoặc cpu)')
    
    args = parser.parse_args()
    
    # Nếu không có tùy chọn nào được chọn, mặc định là run_all
    if not (args.run_all or args.preprocess or args.check_model or args.train):
        args.run_all = True
    
    main(args)
