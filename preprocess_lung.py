import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
from tensorflow.keras.preprocessing import image
from scipy.signal import butter, filtfilt, iirnotch

# Filter definitions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype='band')

def apply_bandpass_filter(data, sr, lowcut=100, highcut=2000):
    b, a = butter_bandpass(lowcut, highcut, sr)
    return filtfilt(b, a, data)

def apply_notch_filter(data, sr, freq=50.0, Q=30.0):
    b, a = iirnotch(freq / (0.5 * sr), Q)
    return filtfilt(b, a, data)

def pad_or_truncate(y, sr, target_len_sec=4):
    target_len = int(sr * target_len_sec)
    if len(y) > target_len:
        return y[:target_len]
    else:
        return np.pad(y, (0, target_len - len(y)))

# ✅ Preprocessing Function
def preprocess_lung_audio_to_array(filepath):
    y, sr = librosa.load(filepath, sr=None)
    if np.max(np.abs(y)) < 0.01 or len(y) / sr < 2:
        raise ValueError("Bad lung audio signal")

    y = y / np.max(np.abs(y))  # Normalize
    y = apply_notch_filter(y, sr)
    y = apply_bandpass_filter(y, sr)
    y = librosa.resample(y, orig_sr=sr, target_sr=8000)
    sr = 8000
    y = pad_or_truncate(y, sr, target_len_sec=4)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip(mel_db, -60, 0)
    mel_db = (mel_db + 60) / 60  # Normalize to 0–1

    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)  # ✅ 128x128 px image
    plt.axis('off')
    librosa.display.specshow(mel_db, sr=sr, fmax=2000, cmap='inferno')
    temp_image_path = f"{uuid.uuid4().hex}.png"
    fig.savefig(temp_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = image.load_img(temp_image_path, target_size=(128, 128))
    x = image.img_to_array(img) / 255.0
    os.remove(temp_image_path)

    return np.expand_dims(x, axis=0)
