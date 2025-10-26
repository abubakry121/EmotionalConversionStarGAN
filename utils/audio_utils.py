import librosa
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 80

def load_wav(file_path, sr=SR):
    wav, _ = librosa.load(file_path, sr=sr, mono=True)
    return wav

def save_wav(wav, file_path, sr=SR):
    sf.write(file_path, wav, sr)

def extract_features(wav, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP):
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel = librosa.power_to_db(mel)

    # Fundamental frequency (F0)
    f0 = librosa.yin(
        wav,
        fmin=50,
        fmax=600,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length
    )
    f0 = np.nan_to_num(f0)  # replace NaN with 0

    # Energy
    energy = np.sum(mel, axis=0)

    return mel, f0, energy

def preprocess_wav(file_path):
    wav = load_wav(file_path)
    mel, f0, energy = extract_features(wav)
    return {
        "mel": mel,
        "f0": f0,
        "energy": energy
    }
