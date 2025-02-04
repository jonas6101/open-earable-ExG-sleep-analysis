import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\EyeMovement'
filtered_file_path = folder_path + r'\filtered_data.csv'
df = pd.read_csv(filtered_file_path)

eeg_data = df['filtered_data'].values
sf = 236
duration = len(eeg_data) / sf
time = np.linspace(0, duration, len(eeg_data))

downsample_factor = 10  # aadjust this value as needed
eeg_data_downsampled = eeg_data[::downsample_factor]
time_downsampled = time[::downsample_factor]

# Plot the EEG data in the time domain
plt.figure(figsize=(12, 6))
plt.plot(time_downsampled, eeg_data_downsampled, color='blue', linewidth=0.8)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (µV)', fontsize=12)
plt.title('EEG Data in Time Domain (Downsampled)', fontsize=14, weight='bold')
plt.ylim(-250, 250)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

output_file_path = folder_path + '/eeg_time_domain_downsampled.png'
plt.savefig(output_file_path, dpi=300)
plt.show()

print(f"Time-domain plot saved at: {output_file_path}")


