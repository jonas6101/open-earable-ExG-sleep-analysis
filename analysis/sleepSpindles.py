import os
import matplotlib
import numpy as np
import pandas as pd
import yasa

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P011\250111\rawData'
filtered_file_path = folder_path + '/eeg_with_sleep_stages.csv'
df = pd.read_csv(filtered_file_path)

# Extract EEG data and sleep stages
eeg_data = df['filtered_data'].values
sleep_stages = df['sleep_stage'].values  # Apple's sleep stages
sf = 241  # Sampling frequency in Hz

# Map Apple sleep stages to integers (YASA-compatible)
stage_mapping = {
    'Awake': 0,
    'Core': 2,  # Assuming Core corresponds to N2
    'Deep': 3,  # Map to N3
    'REM': 4  # Map to REM
}
sleep_stages_int = np.array([stage_mapping.get(stage, -1) for stage in sleep_stages])  # -1 for unknown

# Filter out unknown sleep stages
valid_idx = sleep_stages_int != -1
eeg_data = eeg_data[valid_idx]
sleep_stages_int = sleep_stages_int[valid_idx]

# Detect sleep spindles
sp = yasa.spindles_detect(eeg_data, sf=sf, hypno=sleep_stages_int, include=(1, 2, 3))

spindle_df = sp.summary()

output_dir = os.path.join(folder_path, 'spindle_figures')
os.makedirs(output_dir, exist_ok=True)

padding = int(5 * sf)  # 5 seconds before and after in samples
for i, spindle in spindle_df.iterrows():
    start_idx = int(spindle['Start'] * sf)  # Convert start time to index
    end_idx = int(spindle['End'] * sf)  # Convert end time to index

    window_start = max(0, start_idx - padding)
    window_end = min(len(eeg_data), end_idx + padding)

    signal_window = eeg_data[window_start:window_end]
    mask_window = sp.get_mask()[window_start:window_end]

    # Highlight all spindles in the window
    spindles_highlight = signal_window * mask_window
    spindles_highlight[spindles_highlight == 0] = np.nan  # Replace non-spindle samples with NaN

    time_window = np.arange(len(signal_window)) / sf  # Reset the time vector to start at 0

    plt.figure(figsize=(14, 4))
    plt.plot(time_window, signal_window, 'k', label="EEG Signal")  # Original signal in black
    plt.plot(time_window, spindles_highlight, 'indianred', label="Spindles")  # Highlighted spindles in red
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Spindle {i + 1}")
    plt.xlim([0, time_window[-1]])  # Ensure the x-axis starts at 0
    plt.ylim(-100, 100)
    plt.legend()
    sns.despine()

    file_name = os.path.join(output_dir, f'spindle_{i + 1}.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

print(f"Figures saved in {output_dir}")
