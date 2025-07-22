import os
import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from skimage.transform import resize
 
# === CONFIG ===

sr_target = 8000

segment_duration = 2.5  # seconds

n_mels = 128

fmax = 400

img_size = (256, 256)
 
# === Bandpass filter ===

def butter_bandpass(lowcut=20, highcut=200, fs=8000, order=4):

    nyq = 0.5 * fs

    low = lowcut / nyq

    high = highcut / nyq

    return butter(order, [low, high], btype='band')
 
def preprocess_audio_to_array(file_path):

    y, sr = librosa.load(file_path, sr=None)

    y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)

    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
 
    b, a = butter_bandpass()

    y_filtered = filtfilt(b, a, y)

    y_clean = y_filtered[int(2.0 * sr_target):]  # Skip initial stethoscope noise
 
    seg_len = int(segment_duration * sr_target)

    total_segments = len(y_clean) // seg_len

    segments = [y_clean[i*seg_len:(i+1)*seg_len] for i in range(total_segments)]
 
    specs = []

    for seg in segments:

        mel = librosa.feature.melspectrogram(y=seg, sr=sr_target, n_mels=n_mels, fmax=fmax)

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_norm = np.clip((mel_db + 60) / 60, 0, 1)

        mel_resized = resize(mel_norm, img_size, preserve_range=True, anti_aliasing=True)

        mel_rgb = np.stack([mel_resized]*3, axis=-1)

        specs.append(mel_rgb)
 
    return np.array(specs)

 

# import os
# import librosa
# import librosa.display
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import uuid
# from scipy.signal import butter, filtfilt, iirnotch
# from tensorflow.keras.preprocessing import image
# from skimage.transform import resize

# # Configuration
# TARGET_SR = 8000
# DURATION = 2.5
# N_MELS = 128
# FMAX = 400
# IMG_SIZE = (256, 256)

# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def apply_bandpass_filter(data, sr, lowcut=20, highcut=200):
#     b, a = butter_bandpass(lowcut, highcut, sr)
#     return filtfilt(b, a, data)

# def apply_notch_filter(data, sr, freq=50.0, Q=30.0):
#     b, a = iirnotch(freq / (0.5 * sr), Q)
#     return filtfilt(b, a, data)

# def resample_audio(y, orig_sr, target_sr=8000):
#     return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr), target_sr

# def pad_or_truncate(y, sr, target_len=2.5):
#     target_samples = int(target_len * sr)
#     if len(y) > target_samples:
#         return y[:target_samples]
#     else:
#         return np.pad(y, (0, target_samples - len(y)), mode='constant')


# def preprocess_audio_to_array(filepath):
#     print("[DEBUG] Preprocess file loaded from:", __file__)
#     y, sr = librosa.load(filepath, sr=None)
#     y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

#     print("[DEBUG] max abs:", np.max(np.abs(y)))
#     print("[DEBUG] is finite:", np.all(np.isfinite(y)))
#     print("[DEBUG] length in sec:", len(y) / sr)

#     # Basic checks
#     if y is None or len(y) == 0:
#         raise ValueError("Empty audio signal")

#     y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
#     if np.max(np.abs(y)) < 1e-8 or len(y) / sr < 1.5:
#         raise ValueError("Bad audio signal")

#     # Normalize
#     y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

#     # Apply filtering
#     y = apply_notch_filter(y, sr)
#     y = apply_bandpass_filter(y, sr)

#     # Resample
#     y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
#     sr = TARGET_SR

#     # Remove initial 2s of noise
#     y = y[int(2.0 * sr):]

#     # Segment
#     segment_len = int(DURATION * sr)
#     total_segments = len(y) // segment_len
#     if total_segments == 0:
#         raise ValueError("Not enough audio to form even one segment")

#     segments = [y[i * segment_len:(i + 1) * segment_len] for i in range(total_segments)]

#     segment_images = []
#     for seg in segments:
#         mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=N_MELS, fmax=FMAX)
#         mel_db = librosa.power_to_db(mel, ref=np.max)
#         mel_norm = np.clip((mel_db + 60) / 60, 0, 1)
#         mel_resized = resize(mel_norm, IMG_SIZE, preserve_range=True, anti_aliasing=True)
#         mel_rgb = np.stack([mel_resized] * 3, axis=-1)  # Convert to 3-channel RGB
#         segment_images.append(mel_rgb)

#     x = np.array(segment_images)  # shape: (segments, 256, 256, 3)
#     x = x.astype(np.float32)
#     return x

