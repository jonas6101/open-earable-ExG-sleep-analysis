import numpy as np
import pandas as pd
import scipy.signal
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define
sos = scipy.signal.iirfilter(2, Wn=[1, 30], fs=256, btype="bandpass", ftype="butter", output="sos")
inamp_gain = 50

# Load the unpacked data
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\NoiseTest\2'
file_path = folder_path + '/combined_data.csv'
df = pd.read_csv(file_path)

# Convert 'timestamp' column to datetime and then extract the time component
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f').dt.time

# Extract the raw data and apply the bandpass filter to the entire dataset
yraw = df['rawData']
df['filtered_data'] = scipy.signal.sosfilt(sos, yraw)
df['filtered_data'] = (df['filtered_data'] / inamp_gain) * 1e6  # Conversion to microvolts

# Save the filtered data to a new CSV file with only the time in 'timestamp'
filtered_file_path = folder_path + '/filtered_data.csv'
df.to_csv(filtered_file_path, index=False)

print(f"Filtered data saved at: {filtered_file_path}")
