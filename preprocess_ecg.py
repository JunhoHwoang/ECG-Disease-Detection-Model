import numpy as np
from scipy.signal import resample_poly, filtfilt, butter

# Z-score normalization
def z_score_normalize(x, is_padded, original_len):
    if is_padded:
        valid = x[:original_len]
        mean = np.mean(valid)
        std = np.std(valid) if np.std(valid) != 0 else 1
    else:
        mean = np.mean(x)
        std = np.std(x) if np.std(x) != 0 else 1
    return (x - mean) / std

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_ecg(signal, header, target_fs=400, target_length=4000,
                   expected_leads=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'], default_fs=500):
    signal = np.array(signal)
    fs = header.get('fs', default_fs)
    lead_names = header.get('sig_name', expected_leads)
    n_samples, n_channels = signal.shape

    if fs != target_fs:
        gcd = np.gcd(fs, target_fs)
        up = target_fs // gcd
        down = fs // gcd
        resampled = resample_poly(signal, up, down)
    else:
        resampled = signal
    resample_len = resampled.shape[0]

    reordered = np.zeros((resample_len, len(expected_leads)))
    for i, lead in enumerate(expected_leads):
        if lead in lead_names:
            idx = lead_names.index(lead)
            if idx < n_channels:
                col = resampled[:, idx]
                if np.isnan(col).any():
                    nan_idx = np.isnan(col)
                    not_nan_idx = ~nan_idx
                    col[nan_idx] = np.interp(np.flatnonzero(nan_idx), np.flatnonzero(not_nan_idx), col[not_nan_idx])
                col = bandpass_filter(col, 0.67, 45, fs)
                reordered[:, i] = col

    if resample_len < target_length:
        padded = np.pad(reordered, ((0, target_length - resample_len), (0, 0)), mode='constant')
        is_padded = True
    else:
        start = (resample_len - target_length) // 2
        padded = reordered[start:start+target_length, :]
        is_padded = False

    for i in range(len(expected_leads)):
        padded[:,i] = z_score_normalize(padded[:,i], is_padded, resample_len)

    return padded.astype(np.float32)