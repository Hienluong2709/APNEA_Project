# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display

# # Đường dẫn
# root_dir = 'C:/Users/luong/OneDrive - Thanglele/O D/64KTPM3/NCKH/PSG_AUDIO/data'
# # Đường dẫn thư mục APNEA_EDF chứa các bệnh nhân
# dirname = r'C:\Users\luong\OneDrive - Thanglele\O D\64KTPM3\NCKH\PSG_AUDIO\data\PSG-AUDIO\APNEA_EDF'
# # Đường dẫn thư mục chứa label types
# dirname_types = r'C:\Users\luong\OneDrive - Thanglele\O D\64KTPM3\NCKH\PSG_AUDIO\data\APNEA_types'

# # Danh sách bệnh nhân: duyệt thư mục con
# patients_l = []
# for folder in os.listdir(dirname):
#     subdir = os.path.join(dirname, folder)
#     if os.path.isdir(subdir):
#         files = os.listdir(subdir)
#         if any(f.endswith('_ap.npy') for f in files):
#             patients_l.append(folder)

# print("Bệnh nhân hợp lệ:", patients_l)


# def return_dir_(_patient, apnea=True):
#     subdir = os.path.join(dirname, _patient)
#     filename = f"{_patient}" + ('_' if apnea else '_n') + "ap.npy"
#     full_path = os.path.join(subdir, filename)
#     print("Đang load:", full_path)
#     return np.load(full_path)

# def read_types(_patient):
#     full_path = os.path.join(dirname_types, f"{_patient}_ap_types.npy")
#     print("Đang load:", full_path)
#     return np.load(full_path)

# def show_spectrogram(y, ax):
#     sr = 16000
#     D = np.abs(librosa.stft(y))
#     DB = librosa.amplitude_to_db(D, ref=np.max)
#     librosa.display.specshow(DB, sr=sr, x_axis="time", y_axis="log", cmap="magma", ax=ax)
#     ax.set_title("Spectrogram")

# def show_mel_spec(y, ax):
#     sr = 16000
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#     librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", cmap="inferno", ax=ax)
#     ax.set_title("Mel Spectrogram")

# # Ánh xạ loại apnea
# map_type = {1:'Hypopnea', 2:'MixedApnea', 3:'ObstructiveApnea', 4:'CentralApnea'}

# # Load bệnh nhân đầu tiên
# _patient = patients_l[0]
# ap_arr = return_dir_(_patient, apnea=True).astype("float32")
# nap_arr = return_dir_(_patient, apnea=False).astype("float32")
# ap_types = read_types(_patient)


# # Vẽ ví dụ
# i = 160
# f, ax = plt.subplots(ncols=2, nrows=3, figsize=(12,10))

# ax[0][0].plot(ap_arr[i], lw=0.5)
# ax[0][0].axis("off")
# ax[0][0].set_title(f"Apnea: {map_type[ap_types[i]]}")
# show_spectrogram(ap_arr[i], ax[1][0])
# show_mel_spec(ap_arr[i], ax[2][0])

# ax[0][1].plot(nap_arr[i], lw=0.5)
# ax[0][1].axis("off")
# ax[0][1].set_title("Non-Apnea")
# show_spectrogram(nap_arr[i], ax[1][1])
# show_mel_spec(nap_arr[i], ax[2][1])

# plt.tight_layout()
# plt.show()


# from utils.evaluate_AHI import calculate_ahi_from_y_blocks_dir

# block_dir = "data/blocks"
# ahi, apnea_count, total_blocks = calculate_ahi_from_y_blocks_dir(block_dir)

# print(f"✅ AHI: {ahi:.2f}")
# print(f"Apnea blocks: {apnea_count} / {total_blocks}")

# import numpy as np

# patient_id = "00000995-100507"
# base_path = f"data/PSG-AUDIO/APNEA_EDF/{patient_id}"

# ap_path = f"{base_path}/{patient_id}_ap.npy"
# nap_path = f"{base_path}/{patient_id}_nap.npy"

# ap = np.load(ap_path, allow_pickle=True)
# nap = np.load(nap_path, allow_pickle=True)

# print("✅ Loaded successfully!")
# print("ap shape:", ap.shape)
# print("nap shape:", nap.shape)
# print("1 sample ap block:", ap[0].shape)
# print("1 sample nap block:", nap[0].shape)





