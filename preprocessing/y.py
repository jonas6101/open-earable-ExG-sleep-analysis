import matplotlib
import numpy as np
import pandas as pd
import yasa
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load the data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P009\250118\rawData'
filtered_file_path = folder_path + '/eeg_with_sleep_stages.csv'
df = pd.read_csv(filtered_file_path)

# Extract EEG data and sleep stages
eeg_data = df['filtered_data'].values
sleep_stages = df['sleep_stage'].values  # Apple's sleep stages
sf = 241# Sampling frequency in Hz

# Map Apple sleep stages to integers (YASA-compatible)
stage_mapping = {
    'Awake': 0,
    'Core': 1,   # Map to a stage
    'Deep': 2,   # Map to N3
    'REM': 3    # Map to REM
}
sleep_stages_int = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])  # -1 for unknown

# Generate the spectrogram with YASA
fig = yasa.plot_spectrogram(
    data=eeg_data,
    sf=sf,
    win_sec= 30,
    fmax=100,
    cmap='jet'
)

# Save the figure
output_file_path = folder_path + '/spectrogram_with_apple_hypnogram.png'
fig.savefig(output_file_path, dpi=300)
plt.close(fig)

print(f"Spectrogram with Apple sleep stages saved at: {output_file_path}")