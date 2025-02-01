
import matplotlib
import numpy as np
import pandas as pd
import mne
import yasa
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P003\241212\rawData'
filtered_file_path = folder_path + '/eeg_with_sleep_stages.csv'
df = pd.read_csv(filtered_file_path)

# Extract EEG data, Apple Watch sleep stages, and sampling frequency
eeg_data = df['filtered_data'].values  # 1D array of EEG signal
apple_watch_stages = df['sleep_stage'].values  # Sleep stages from Apple Watch
sf = 241  # Sampling frequency in Hz

# Map Apple sleep stages to integers
stage_mapping = {
    'Awake': 0,
    'Core': 2,  # Assuming Core corresponds to N2
    'Deep': 3,  # Map to N3
    'REM': 4    # Map to REM
}
ground_truth_hypno = np.array([stage_mapping.get(stage, -1) for stage in apple_watch_stages])  # -1 for unknown

# Filter out unknown sleep stages
valid_idx = ground_truth_hypno != -1
eeg_data = eeg_data[valid_idx]
ground_truth_hypno = ground_truth_hypno[valid_idx]

# Reshape data to (n_channels, n_samples) for MNE
eeg_data = eeg_data[np.newaxis, :]  # Add a channel dimension: (1, n_samples)

# Create MNE info object
info = mne.create_info(ch_names=['EEG'], sfreq=sf, ch_types=['eeg'])

# Create MNE Raw object
raw = mne.io.RawArray(eeg_data, info)

# Automatic sleep staging using YASA
sls = yasa.SleepStaging(raw, eeg_name='EEG')  # Initialize sleep staging
hypno_pred = sls.predict()  # Predict sleep stages
hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W", "N1", etc., to integers

# Plot YASA-predicted hypnogram
plt.figure(figsize=(10, 4))
yasa.plot_hypnogram(hypno_pred)
plt.show()

# Calculate the agreement
agreement = accuracy_score(ground_truth_hypno, hypno_pred[:len(ground_truth_hypno)])  # Ensure equal lengths
print(f"The accuracy between YASA and Apple Watch is {100 * agreement:.3f}%")

# Save the results to CSV (optional)
output_path = folder_path + '/comparison_hypnogram.csv'
comparison_df = pd.DataFrame({
    'YASA_Sleep_Stages': hypno_pred[:len(ground_truth_hypno)],
    'Apple_Watch_Sleep_Stages': ground_truth_hypno
})
comparison_df.to_csv(output_path, index=False)
print(f"Comparison results saved to {output_path}")
