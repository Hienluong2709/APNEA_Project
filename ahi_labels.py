import os
import numpy as np
import pandas as pd

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

# 📌 Chạy script
if __name__ == "__main__":
    generate_ahi_labels("data/PSG-AUDIO/APNEA_EDF", default_duration_sec=6*3600)  # 6 tiếng mặc định
