import numpy as np
import librosa

def extract_melspectrum(y, sr=16000):
    """
    Tính Mel-spectrum (không log) từ tín hiệu âm thanh.
    Đầu ra: ma trận 64 x 684 đã cắt hoặc pad.
    """
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
