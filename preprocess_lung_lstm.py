import os
import librosa
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

#CONFIG
SAMPLE_RATE = 8000
DURATION = 2.5 
N_MFCC = 40
LOWCUT = 20
HIGHCUT = 2000
NOTCH_FREQ = 50.0
QUALITY = 30.0

#Filters
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass(data, fs):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs)
    return filtfilt(b, a, data)

def apply_notch(data, fs):
    b_notch, a_notch = iirnotch(NOTCH_FREQ, QUALITY, fs)
    return filtfilt(b_notch, a_notch, data)




from scipy.io.wavfile import write

 
def preprocess_lung_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y = apply_bandpass(y, sr)
    y = apply_notch(y, sr)
    
    gain = 3
    y_amplified = y * gain
    y_amplified = np.clip(y_amplified, -1.0, 1.0)
    filtered_path = file_path.replace(".wav", "_filtered.wav")
    write(filtered_path, SAMPLE_RATE, (y_amplified * 32767).astype(np.int16))

    segment_samples = int(DURATION * sr)
    segments = []

    
    for start in range(0, len(y) - segment_samples + 1, segment_samples):
        segment = y[start:start + segment_samples]
        segment *= np.hamming(len(segment))

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, lifter=22)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        full_features = np.vstack([mfcc, delta, delta2])
        full_features = full_features.T 
        full_features = full_features.T  
        full_features = np.reshape(full_features, (120, -1))
        full_features = np.transpose(full_features, (1, 0))  

        segments.append(full_features)

    
    segments = np.array(segments) 


    padded_segments = []
    for seg in segments:
        if seg.shape[0] < 63:
            pad_len = 63 - seg.shape[0]
            pad = np.zeros((pad_len, 120))
            padded_seg = np.vstack([seg, pad])
        else:
            padded_seg = seg[:63, :]  

        padded_segments.append(padded_seg)

    padded_segments = np.array(padded_segments)  
    print("Final lung shape:", padded_segments.shape)
    return padded_segments, filtered_path

    
