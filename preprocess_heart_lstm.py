import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, iirnotch

#CONFIGURATION
SAMPLE_RATE = 8000
SEGMENT_DURATION = 2.0 
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
N_MFCC = 40
MAX_FRAMES = 96

#FILTERS
def bandpass_filter(signal, sr, lowcut=20, highcut=200):
    nyq = 0.5 * sr
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, sr, notch_freq=50.0, Q=30.0):
    nyq = 0.5 * sr
    b, a = iirnotch(notch_freq / nyq, Q)
    return filtfilt(b, a, signal)



#QUALITY CHECKS
def is_clipped(y, threshold=0.98):
    return np.max(np.abs(y)) > threshold

def is_too_silent(y, rms_thresh=0.005):
    return np.sqrt(np.mean(y**2)) < rms_thresh

def is_spectrally_flat(y, flat_thresh=0.3):
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    return np.mean(flatness) > flat_thresh


from scipy.io.wavfile import write



#MAIN FUNCTION
def preprocess_heart_lstm_fn(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    #filters
    y = bandpass_filter(y, sr)
    y = notch_filter(y, sr)

    if is_clipped(y) or is_too_silent(y) or is_spectrally_flat(y):
        raise ValueError("Audio failed quality checks.")

    #Amplify of ,wav audio we got
    gain = 3
    y_amplified = y * gain
    y_amplified = np.clip(y_amplified, -1.0, 1.0)
    filtered_path = path.replace(".wav", "_filtered.wav")
    write(filtered_path, SAMPLE_RATE, (y_amplified * 32767).astype(np.int16))


    # Segmention of audio
    segments = [y[i:i + SEGMENT_SAMPLES] for i in range(0, len(y) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES)]
    processed_segments = []

    for segment in segments:
        if len(segment) < SEGMENT_SAMPLES:
            segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)))

        mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        
        if mfcc.shape[1] < MAX_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
        elif mfcc.shape[1] > MAX_FRAMES:
            mfcc = mfcc[:, :MAX_FRAMES]

        mfcc = mfcc.T 
        if mfcc.shape != (96, 40):
            print("[ERROR] MFCC shape mismatch:", mfcc.shape)
            continue  

        processed_segments.append(mfcc)

    output = np.array(processed_segments).astype(np.float32)
    print("[DEBUG] Final input shape to model:", output.shape)  
    return output, filtered_path




