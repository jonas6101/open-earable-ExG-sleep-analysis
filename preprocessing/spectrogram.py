import os
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

#!obsolete use yasa script!

# Folder paths
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\workingMLdata'
combined_file_path = folder_path + '/filtered_and_apple_data.csv'
output_folder = folder_path + '/epoch_spectrograms'
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(combined_file_path)

# parameters for epoching and spectrogram generation
sampling_rate = 250  # Hz
epoch_duration = 30  # seconds
epoch_samples = epoch_duration * sampling_rate
num_epochs = len(df) // epoch_samples

for epoch in range(num_epochs):
    start = epoch * epoch_samples
    end = start + epoch_samples
    epoch_data = df['filtered_data'][start:end]

    current_sleep_stage = df['sleep_stage'][start:end].mode()[0]

    frequencies, times, Sxx = scipy.signal.spectrogram(epoch_data, fs=sampling_rate, nperseg=512, noverlap=256)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='jet', vmin=-100, vmax=50)
    plt.ylim(0, 40)  # Limit the frequency range to typical EEG frequencies
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.title(f'Spectrogram for Epoch {epoch + 1} (30 seconds) - Sleep Stage: {current_sleep_stage}')
    plt.colorbar(label='Power (dB)')

    output_path = os.path.join(output_folder, f'spectrogram_epoch_{epoch + 1}.png')
    plt.savefig(output_path)
    plt.close()

print(f"Spectrograms saved to {output_folder}")