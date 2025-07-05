import os
import numpy as np

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
