import os
import pandas as pd
import numpy as np
from itertools import groupby
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def extract_features_from_preds(preds):
    n_total = len(preds)
    n_1 = sum(preds)

    blocks = [(key, sum(1 for _ in group)) for key, group in groupby(preds)]
    one_blocks = [length for value, length in blocks if value == 1]
    zero_blocks = [length for value, length in blocks if value == 0]

    num_1_blocks = len(one_blocks)
    avg_len_1 = np.mean(one_blocks) if one_blocks else 0
    avg_len_0 = np.mean(zero_blocks) if zero_blocks else 0

    return [n_1, num_1_blocks, avg_len_1, avg_len_0, n_total]


def load_ahi_labels(path="ahi_labels2.csv"):
    df = pd.read_csv(path)
    return dict(zip(df["patient_id"], df["ahi_psg"]))


def prepare_dataset(pred_dir, ahi_dict):
    X, y, patient_ids = [], [], []

    for fname in os.listdir(pred_dir):
        if not fname.endswith(".csv"):
            continue

        patient_id = fname.split("_preds")[0]
        if patient_id not in ahi_dict:
            continue

        df = pd.read_csv(os.path.join(pred_dir, fname))
        preds = df["label_pred"].tolist()
        features = extract_features_from_preds(preds)

        X.append(features)
        y.append(ahi_dict[patient_id])
        patient_ids.append(patient_id)

    return np.array(X), np.array(y), patient_ids


def classify_osa(ahi):
    if ahi < 5:
        return "Không"
    elif ahi < 15:
        return "Nhẹ"
    elif ahi < 30:
        return "Vừa"
    else:
        return "Nặng"


def train_and_evaluate_ahi_regression(pred_dir, ahi_csv="ahi_labels2.csv"):
    ahi_dict = load_ahi_labels(ahi_csv)
    X, y, patient_ids = prepare_dataset(pred_dir, ahi_dict)

    model = RANSACRegressor()
    model.fit(X, y)
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    pcc, _ = pearsonr(y, y_pred)

    print("\n📈 Đánh giá ước lượng AHI:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  PCC:  {pcc:.3f}")

    df_result = pd.DataFrame({
        "patient_id": patient_ids,
        "ahi_psg": y,
        "ahi_pred": y_pred
    })
    df_result["error"] = df_result["ahi_pred"] - df_result["ahi_psg"]
    df_result["osa_pred_level"] = df_result["ahi_pred"].apply(classify_osa)

    df_result.to_csv("ahi_estimation_results_trans.csv", index=False)
    print("\n✅ Đã lưu kết quả vào ahi_estimation_results_trans.csv")

    return model


if __name__ == "__main__":
    pred_dir = "predictions_trans"               
    ahi_csv_path = "ahi_labels2.csv"        

    train_and_evaluate_ahi_regression(pred_dir, ahi_csv_path)
