import os
import pandas as pd
import numpy as np

def estimate_ahi(ap_path, duration_sec):
    events = np.load(ap_path)
    num_events = len(events)
    duration_hours = duration_sec / 3600
    ahi = num_events / duration_hours
    return num_events, duration_hours, ahi

def generate_ahi_labels(apnea_edf_dir, default_duration_sec=6*3600):
    records = []

    for patient_id in os.listdir(apnea_edf_dir):
        patient_dir = os.path.join(apnea_edf_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        ap_files = [f for f in os.listdir(patient_dir) if f.endswith('_ap.npy')]
        if not ap_files:
            print(f"⚠️ Không tìm thấy file *_ap.npy trong {patient_id}")
            continue

        ap_path = os.path.join(patient_dir, ap_files[0])
        num_events, duration_hours, ahi = estimate_ahi(ap_path, default_duration_sec)

        print(f"✅ {patient_id}: {num_events} events, {duration_hours:.2f}h → AHI = {ahi:.2f}")
        records.append({
            "patient_id": patient_id,
            "num_events": num_events,
            "duration_hours": duration_hours,
            "ahi_psg": ahi
        })

    df = pd.DataFrame(records)
    df.to_csv("ahi_labels.csv", index=False)
    print("\n📁 Đã lưu file: ahi_labels.csv")

def extract_features_from_predictions(pred_dir, ahi_labels_path):
    ahi_df = pd.read_csv(ahi_labels_path)
    feature_rows = []

    for file in os.listdir(pred_dir):
        if not file.endswith("_preds.csv"):
            continue

        patient_id = file.replace("_preds.csv", "")
        pred_path = os.path.join(pred_dir, file)
        df = pd.read_csv(pred_path)

        preds = df["label_pred"].tolist()
        num_segments = len(preds)
        num_1 = sum(preds)

        # Tính số block 1 liên tiếp
        num_1_blocks, len_1_blocks, len_0_blocks = 0, [], []
        i = 0
        while i < len(preds):
            current = preds[i]
            length = 1
            while i + 1 < len(preds) and preds[i + 1] == current:
                i += 1
                length += 1
            if current == 1:
                num_1_blocks += 1
                len_1_blocks.append(length)
            else:
                len_0_blocks.append(length)
            i += 1

        avg_len_1 = np.mean(len_1_blocks) if len_1_blocks else 0
        avg_len_0 = np.mean(len_0_blocks) if len_0_blocks else 0

        feature_rows.append({
            "patient_id": patient_id,
            "num_segments": num_segments,
            "num_1": num_1,
            "num_1_blocks": num_1_blocks,
            "avg_len_1": avg_len_1,
            "avg_len_0": avg_len_0
        })

    features_df = pd.DataFrame(feature_rows)
    merged_df = pd.merge(ahi_df, features_df, on="patient_id", how="inner")

    # Tạo cột is_val chia 80-20 theo thứ tự bệnh nhân
    merged_df = merged_df.sort_values("patient_id").reset_index(drop=True)
    num_val = int(0.2 * len(merged_df))
    merged_df["is_val"] = 0
    merged_df.loc[:num_val - 1, "is_val"] = 1

    merged_df.to_csv("ahi_labels.csv", index=False)
    print("✅ Đã cập nhật và lưu file: ahi_labels.csv")

if __name__ == "__main__":
    generate_ahi_labels("data/PSG-AUDIO/APNEA_EDF", default_duration_sec=6*3600)  # 6 tiếng mặc định
    extract_features_from_predictions("prediction", "ahi_labels.csv")
