import os
import numpy as np
import pandas as pd
import yasa
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Load the data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241204\rawData'
filtered_file_path = os.path.join(folder_path, 'filtered_data.csv')
df = pd.read_csv(filtered_file_path)

# Filter for the first hour (assuming a 'time' column in seconds or as datetime)
# If 'time' is in datetime format, convert and filter:
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Adjust if already in datetime format
start_time = df['timestamp'].min()
end_time = start_time + pd.Timedelta(hours=2.5)
df_first_hour = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

# Extract EEG data for the first hour
eeg_data = df['filtered_data'].values
sf = 250  # Sampling frequency in Hz

# Generate the spectrogram with YASA
fig = yasa.plot_spectrogram(
    data=eeg_data,
    sf=sf,
    win_sec=30,
    fmax=30,
    cmap='jet'
)

# Save the figure
output_file_path = os.path.join(folder_path, 'spectrogram_noise.png')
fig.savefig(output_file_path, dpi=300)
plt.close(fig)

print(f"Spectrogram saved at: {output_file_path}")
