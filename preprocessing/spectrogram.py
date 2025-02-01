import os
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define folder paths
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\workingMLdata'
combined_file_path = folder_path + '/filtered_and_apple_data.csv'
output_folder = folder_path + '/epoch_spectrograms'
os.makedirs(output_folder, exist_ok=True)

# Load the combined data with sleep stages
df = pd.read_csv(combined_file_path)

# Define parameters for epoching and spectrogram generation
sampling_rate = 250  # Hz
epoch_duration = 30  # seconds
epoch_samples = epoch_duration * sampling_rate
num_epochs = len(df) // epoch_samples

# Generate and save spectrograms for each 30-second epoch
for epoch in range(num_epochs):
    # Extract data for the current epoch
    start = epoch * epoch_samples
    end = start + epoch_samples
    epoch_data = df['filtered_data'][start:end]

    # Get the sleep stage for the current epoch (assumes one stage per epoch)
    current_sleep_stage = df['sleep_stage'][start:end].mode()[0]  # Using mode to handle any minor stage variations within the epoch

    # Calculate the spectrogram for the epoch
    frequencies, times, Sxx = scipy.signal.spectrogram(epoch_data, fs=sampling_rate, nperseg=512, noverlap=256)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='jet', vmin=-100, vmax=50)
    plt.ylim(0, 40)  # Limit the frequency range to typical EEG frequencies
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.title(f'Spectrogram for Epoch {epoch + 1} (30 seconds) - Sleep Stage: {current_sleep_stage}')
    plt.colorbar(label='Power (dB)')

    # Save the plot to the specified folder
    output_path = os.path.join(output_folder, f'spectrogram_epoch_{epoch + 1}.png')
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

print(f"Spectrograms saved to {output_folder}")