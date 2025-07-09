import os
import numpy as np
import gc
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Xử lý từng bệnh nhân từ thư mục block
def process_patient_from_blocks(patient_dir):
    x_files = sorted([f for f in os.listdir(patient_dir) if f.startswith("X_")])
    y_files = sorted([f for f in os.listdir(patient_dir) if f.startswith("y_")])

    total_apnea = 0
    total_blocks = 0
    all_energies = []

    for x_f, y_f in zip(x_files, y_files):
        x_path = os.path.join(patient_dir, x_f)
        y_path = os.path.join(patient_dir, y_f)

        try:
            x_data = np.load(x_path, mmap_mode="r")
            y_data = np.load(y_path, mmap_mode="r")
        except Exception as e:
            print(f"❌ Không thể đọc {x_f} hoặc {y_f}: {e}")
            continue

        if len(x_data) == 0 or len(y_data) == 0 or len(x_data) != len(y_data):
            continue

        # Tính năng lượng trung bình của từng block
        mean_energy = np.mean(x_data)  # do đã mel-spec sẵn
        all_energies.append(mean_energy)

        total_apnea += np.sum(y_data)
        total_blocks += len(y_data)

    if total_blocks == 0 or not all_energies:
        return None

    ahi = total_apnea / ((total_blocks * 30) / 3600)  # block 30s -> giờ ngủ
    avg_energy = np.mean(all_energies)

    return avg_energy, ahi

# Load dataset từ thư mục blocks/
def load_dataset(blocks_dir):
    X, y = [], []
    patients = [f for f in os.listdir(blocks_dir) if os.path.isdir(os.path.join(blocks_dir, f))]
    print(f"🔍 Tìm thấy {len(patients)} bệnh nhân")

    valid = 0
    for i, pid in enumerate(patients):
        result = process_patient_from_blocks(os.path.join(blocks_dir, pid))
        if result:
            feature, ahi = result
            X.append([feature])
            y.append(ahi)
            valid += 1
        if (i + 1) % 10 == 0 or i == len(patients) - 1:
            print(f"✅ Đã xử lý {i+1}/{len(patients)} bệnh nhân | Hợp lệ: {valid}")
        gc.collect()
    return np.array(X), np.array(y)

# Huấn luyện và đánh giá
def train_and_evaluate_model(X, y):
    if len(X) < 2:
        print("❌ Không đủ dữ liệu để huấn luyện!")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RANSACRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    return model, y_pred

# Chạy chương trình chính
if __name__ == "__main__":
    BLOCKS_DIR = "data/blocks"
    X, y = load_dataset(BLOCKS_DIR)
    if len(X) == 0:
        print("❌ Không có dữ liệu đầu vào hợp lệ!")
    else:
        model, y_pred = train_and_evaluate_model(X, y)
