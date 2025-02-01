import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

# Ensure the correct backend is used for plotting
matplotlib.use('TkAgg')

# Load the filtered data with sleep stages
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\workingMLdata'
filtered_file_path = folder_path + '/filtered_and_apple_data.csv'
df = pd.read_csv(filtered_file_path)

# Define parameters for spectrogram generation
sampling_rate = 200  # Hz
nperseg = 256  # Length of each segment for FFT
noverlap = 128  # Number of points to overlap between segments

# Extract the filtered data and sleep stages
filtered_data = df['filtered_data'].values
sleep_stages = df['sleep_stage'].values
times = df['elapsed_seconds'].values  # Ensure elapsed_seconds corresponds to spectrogram times

# Generate the spectrogram
frequencies, times_spec, Sxx = scipy.signal.spectrogram(filtered_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

# Convert times for spectrogram to hours for better readability
times_hours = times_spec / 3600  # Convert seconds to hours
elapsed_hours = times / 3600     # Convert elapsed seconds to hours

# Define sleep stage colors
stage_colors = {
    'Awake': 'red',
    'Core': 'blue',
    'Deep': 'green',
    'REM': 'purple'
}

# Map sleep stages to their corresponding colors
stage_colormap = [stage_colors.get(stage, 'gray') for stage in sleep_stages]

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

# Plot the spectrogram
spectrogram = ax1.pcolormesh(times_hours, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='jet', vmin=-100, vmax=50)
ax1.set_ylim(0, 40)  # Limit frequency range to typical EEG frequencies
ax1.set_ylabel('Frequency (Hz)')
ax1.set_title('Spectrogram of Entire Night EEG Recording with Sleep Stages')
cbar = plt.colorbar(spectrogram, ax=ax1, label='Power (dB)')

# Plot the sleep stage bar
bar_width = np.diff(np.append(elapsed_hours, elapsed_hours[-1] + (1 / 3600)))  # Compute width of each bar segment
ax2.bar(elapsed_hours, [1] * len(elapsed_hours), width=bar_width, color=stage_colormap, align='edge', alpha=1)
ax2.set_yticks([])  # Remove y-axis ticks
ax2.set_xlabel('Time (hours)')
ax2.set_title('Sleep Stages')

# Add legend for sleep stages
handles = [plt.Line2D([0], [0], color=color, lw=4, label=stage) for stage, color in stage_colors.items()]
ax1.legend(handles=handles, loc='upper right')

# Adjust layout for better alignment
plt.tight_layout()

# Save the plot to a file
output_file_path = folder_path + '/spectrogram_with_sleep_stages_bar_below.png'
plt.savefig(output_file_path, dpi=300)
plt.close()

print(f"Spectrogram with sleep stages bar saved at: {output_file_path}")