import os
import pandas as pd
import numpy as np
from scipy.signal import welch, find_peaks, iirfilter, sosfiltfilt, cwt, morlet
from scipy.stats import entropy
from datetime import datetime
import zlib

participant_id = "P009"
recording_date = "250118"
epochs_folder = f'../recordings/open-earable-ExG/AppVersion4_BLE/{participant_id}/{recording_date}/epochs'
features_output = f'../ml/features_V3/{participant_id}_{recording_date}_features_V3.csv'

# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 16),
    "beta": (16, 32)
}

# Time-domain features

def zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum() / len(data)

def hjorth_mobility(data):
    diff1 = np.diff(data)
    var_diff1 = np.var(diff1)
    var_data = np.var(data)
    return np.sqrt(var_diff1 / var_data)

def hjorth_complexity(data):
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    mobility = hjorth_mobility(data)
    mobility_diff1 = np.sqrt(np.var(diff2) / np.var(diff1))
    return mobility_diff1 / mobility

# Frequency-domain features

def compute_relative_band_powers(data, sf, bands):
    freqs, psd = welch(data, sf, nperseg=len(data))
    total_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 32)])
    band_powers = {}

    for band_name, band_range in bands.items():
        band_freqs = (freqs >= band_range[0]) & (freqs <= band_range[1])
        absolute_power = np.sum(psd[band_freqs])
        relative_power = absolute_power / total_power if total_power > 0 else 0
        band_powers[f"{band_name}_relative_power"] = relative_power

    # Power ratios
    band_powers["delta_theta_ratio"] = band_powers["delta_relative_power"] / band_powers["theta_relative_power"]
    band_powers["theta_alpha_ratio"] = band_powers["theta_relative_power"] / band_powers["alpha_relative_power"]
    band_powers["alpha_beta_ratio"] = band_powers["alpha_relative_power"] / band_powers["beta_relative_power"]
    band_powers["theta_delta_alpha_beta_ratio"] = (band_powers["theta_relative_power"] + band_powers["delta_relative_power"]) / (band_powers["alpha_relative_power"] + band_powers["beta_relative_power"])

    return band_powers

# CWT-based features
def compute_cwt_features(data, scales, wavelet):
    cwt_matrix = cwt(data, wavelet, scales)
    features = {}

    for i, band in enumerate(BANDS.keys()):
        coeffs = cwt_matrix[i, :]
        features[f"cwt_{band}_percentile_75"] = np.percentile(np.abs(coeffs), 75)
    return features

def compute_spectral_metrics(data, sf):
    freqs, psd = welch(data, sf, nperseg=len(data))

    # Restrict to 0.5–32 Hz range
    valid_range = (freqs >= 0.5) & (freqs <= 32)
    psd = psd[valid_range]
    freqs = freqs[valid_range]

    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]

    # Spectral Edge Frequency (95%)
    edge_freq = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]

    # Median Frequency (50%)
    median_freq = freqs[np.where(cumulative_power >= 0.5 * total_power)[0][0]]

    # Mean Frequency Difference
    mean_freq_diff = edge_freq - median_freq

    # Peak Frequency
    peak_freq = freqs[np.argmax(psd)]

    # Spectral Entropy
    norm_psd = psd / np.sum(psd)
    spec_entropy = entropy(norm_psd)

    return {
        "spectral_edge_frequency": edge_freq,
        "median_frequency": median_freq,
        "mean_frequency_diff": mean_freq_diff,
        "peak_frequency": peak_freq,
        "spectral_entropy": spec_entropy
    }

# Non-linear features
def lempel_ziv_complexity(data):
    binary_sequence = ''.join(['1' if x > np.mean(data) else '0' for x in data])
    return len(zlib.compress(binary_sequence.encode('utf-8')))

# Feature extraction per epoch
def extract_features_and_label(epoch_file, sampling_rate=241):
    df = pd.read_csv(epoch_file)

    if df.empty:
        return None

    data = df['filtered_data'].values
    sleep_stages = df['sleep_stage']
    most_frequent_label = sleep_stages.mode()[0]

    # Time-domain features
    features = {
        "std": np.std(data),
        "variance": np.var(data),
        "skewness": pd.Series(data).skew(),
        "kurtosis": pd.Series(data).kurtosis(),
        "zero_crossing_rate": zero_crossing_rate(data),
        "hjorth_mobility": hjorth_mobility(data),
        "hjorth_complexity": hjorth_complexity(data),
        "percentile_75": np.percentile(data, 75)
    }

    # Frequency-domain features
    relative_band_powers = compute_relative_band_powers(data, sampling_rate, BANDS)
    features.update(relative_band_powers)

    spectral_metrics = compute_spectral_metrics(data, sampling_rate)
    features.update(spectral_metrics)

    # CWT-based features
    scales = np.arange(1, 128)
    features.update(compute_cwt_features(data, scales, morlet))

    features["lempel_ziv_complexity"] = lempel_ziv_complexity(data)

    # Add sleep stage label
    features["sleep_stage_label"] = most_frequent_label

    return features

# Process all epochs
feature_list = []

for file in os.listdir(epochs_folder):
    if file.endswith('.csv'):
        epoch_path = os.path.join(epochs_folder, file)
        features = extract_features_and_label(epoch_path)
        if features:
            epoch_number = ''.join(filter(str.isdigit, file))
            features["epoch"] = epoch_number
            features["participant_id"] = participant_id
            features["date"] = recording_date
            feature_list.append(features)

features_df = pd.DataFrame(feature_list)
features_df.to_csv(features_output, index=False)

print(f"Features saved to: {features_output}")
