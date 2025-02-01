import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Load the data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\EyeMovement'
filtered_file_path = folder_path + r'\filtered_data.csv'
df = pd.read_csv(filtered_file_path)

# Extract EEG data and create a time vector
eeg_data = df['filtered_data'].values
sf = 236  # Sampling frequency in Hz
duration = len(eeg_data) / sf  # Total duration in seconds
time = np.linspace(0, duration, len(eeg_data))  # Time vector

# Downsample the data for better visualization
downsample_factor = 10  # Adjust this value as needed
eeg_data_downsampled = eeg_data[::downsample_factor]
time_downsampled = time[::downsample_factor]

# Plot the EEG data in the time domain
plt.figure(figsize=(12, 6))
plt.plot(time_downsampled, eeg_data_downsampled, color='blue', linewidth=0.8)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (µV)', fontsize=12)  # Adjust unit based on your data
plt.title('EEG Data in Time Domain (Downsampled)', fontsize=14, weight='bold')
plt.ylim(-250, 250)  # Limit the amplitude range
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot to a file
output_file_path = folder_path + '/eeg_time_domain_downsampled.png'
plt.savefig(output_file_path, dpi=300)
plt.show()

print(f"Time-domain plot saved at: {output_file_path}")


