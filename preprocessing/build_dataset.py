import os
import numpy as np
from tqdm import tqdm
import librosa

# ================== Cấu hình ==================
INPUT_DIR = "data"
PSG_DIR = os.path.join(INPUT_DIR, "PSG-AUDIO", "APNEA_EDF")
LABEL_DIR = os.path.join(INPUT_DIR, "APNEA_types")
OUTPUT_DIR = "data/blocks"
BATCH_SIZE = 2000  # Số mẫu mỗi block
SR = 16000         # Sampling rate

# ================== Hàm trích xuất Mel-spectrogram ==================
def extract_melspectrum(y, sr=16000):
    try:
        if len(y) < 1024 or np.isnan(y).any() or np.max(np.abs(y)) < 1e-6:
            return np.zeros((64, 684), dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=499,
            win_length=1024,
            window='blackman',
            n_mels=64,
            fmin=60,
            fmax=22500
        )

        if mel.shape[1] < 684:
            pad = 684 - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
        else:
            mel = mel[:, :684]

        return mel.astype(np.float32)

    except Exception as e:
        print(f"[⚠️ extract_melspectrum error]: {e}")
        return np.zeros((64, 684), dtype=np.float32)

# ================== Hàm lưu block ==================
def save_batch(x, y, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x_batch = np.array(x, dtype=np.float32)
    y_batch = np.array(y, dtype=np.int64)
    np.save(os.path.join(save_dir, f"X_{idx}.npy"), x_batch)
    np.save(os.path.join(save_dir, f"y_{idx}.npy"), y_batch)
    print(f"📂 Đã lưu block {idx} cho bệnh nhân {os.path.basename(save_dir)} | Shape: {x_batch.shape}")
    del x_batch, y_batch

# ================== Xử lý tất cả bệnh nhân ==================
patients = [f for f in os.listdir(PSG_DIR) if os.path.isdir(os.path.join(PSG_DIR, f))]
print(f"📁 Tổng số bệnh nhân: {len(patients)}")

for p in tqdm(patients):
    p_path = os.path.join(PSG_DIR, p)
    ap_path = os.path.join(p_path, f"{p}_ap.npy")
    nap_path = os.path.join(p_path, f"{p}_nap.npy")
    label_path = os.path.join(LABEL_DIR, f"{p}_ap_types.npy")  # chưa dùng nhưng có thể dùng sau

    if not (os.path.exists(ap_path) and os.path.exists(nap_path) and os.path.exists(label_path)):
        print(f"⚠️ Thiếu file cho bệnh nhân {p}, bỏ qua.")
        continue

    try:
        ap_data = np.load(ap_path)
        nap_data = np.load(nap_path)
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc file {p}: {e}")
        continue

    print(f"🔍 Bệnh nhân: {p} | Apnea: {len(ap_data)} | Non-apnea: {len(nap_data)}")

    # Chuẩn bị cho block của từng bệnh nhân
    save_dir = os.path.join(OUTPUT_DIR, p)
    batch_x, batch_y = [], []
    batch_idx = 0

    for seg, label in zip(list(ap_data) + list(nap_data), [1]*len(ap_data) + [0]*len(nap_data)):
        if len(seg) > 10 * SR:
            print(f"⚠️ Bỏ qua đoạn dài ({len(seg)/SR:.1f}s)")
            continue

        mel = extract_melspectrum(seg)
        batch_x.append(mel)
        batch_y.append(label)

        if len(batch_x) >= BATCH_SIZE:
            save_batch(batch_x, batch_y, batch_idx, save_dir)
            batch_x, batch_y = [], []
            batch_idx += 1

    # Lưu phần còn lại nếu có
    if batch_x:
        save_batch(batch_x, batch_y, batch_idx, save_dir)

print("✅ Hoàn thành. Đã xử lý tất cả các block theo từng bệnh nhân.")
