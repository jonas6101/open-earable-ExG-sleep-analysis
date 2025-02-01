import matplotlib
import numpy as np
import pandas as pd
import yasa
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P003\241212\rawData'
filtered_file_path = folder_path + '/eeg_with_sleep_stages.csv'
df = pd.read_csv(filtered_file_path)

# Extract EEG data and sleep stages
eeg_data = df['filtered_data'].values
sleep_stages = df['sleep_stage'].values  # Apple's sleep stages
sf = 241  # Sampling frequency in Hz

# Map Apple sleep stages to integers (YASA-compatible)
stage_mapping = {
    'Awake': 0,
    'Core': 2,   # Assuming Core corresponds to N2
    'Deep': 3,   # Map to N3
    'REM': 4     # Map to REM
}
sleep_stages_int = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])  # -1 for unknown

# Filter out unknown sleep stages
valid_idx = sleep_stages_int != -1
eeg_data = eeg_data[valid_idx]
sleep_stages_int = sleep_stages_int[valid_idx]

# Detect slow waves
sw = yasa.sw_detect(eeg_data, sf=sf, hypno=sleep_stages_int, include=(1, 2, 3))  # Detect SW during N2/N3

# Get slow wave properties
sw_df = sw.summary()

# Select a specific slow wave (e.g., the first one)
selected_sw = sw_df.iloc[45]  # Modify the index as needed
start_idx = int(selected_sw['Start'] * sf)  # Convert start time to index
end_idx = int(selected_sw['End'] * sf)      # Convert end time to index

# Define a 15-second window around the slow wave
padding = int(7.5 * sf)  # 7.5 seconds before and after in samples
window_start = max(0, start_idx - padding)  # Ensure it doesn't go below 0
window_end = min(len(eeg_data), end_idx + padding)  # Ensure it doesn't exceed signal length

# Extract the data and mask for the window
signal_window = eeg_data[window_start:window_end]
mask_window = sw.get_mask()[window_start:window_end]

# Highlight all slow waves in the window
sw_highlight = signal_window * mask_window
sw_highlight[sw_highlight == 0] = np.nan  # Replace non-slow wave samples with NaN

# Create time vector for the window starting at 0
time_window = np.arange(len(signal_window)) / sf  # Reset the time vector to start at 0

# Plot the windowed signal and highlight all slow waves
plt.figure(figsize=(14, 4))
plt.plot(time_window, signal_window, 'k', label="EEG Signal")  # Original signal in black
plt.plot(time_window, sw_highlight, 'dodgerblue', label="Slow Waves")  # Highlighted slow waves in blue
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.xlim([0, time_window[-1]])  # Ensure the x-axis starts at 0
plt.ylim(-200, 200)
plt.title("EEG Signal with Highlighted Slow Waves (15-Second Window)")
plt.legend()
sns.despine()
plt.show()
