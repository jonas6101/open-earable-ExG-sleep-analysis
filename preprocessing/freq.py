import pandas as pd

# Load your data
file_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241204\rawData\eeg_with_sleep_stages.csv'
df = pd.read_csv(file_path)

# Convert 'timestamp' column to datetime format
try:
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f', errors='coerce')
except ValueError as e:
    print("Error parsing timestamps:", e)

# Drop rows with NaT in the 'timestamp' column after parsing
df = df.dropna(subset=['time'])

# Calculate the time differences between consecutive samples
time_diffs = df['time'].diff().dt.total_seconds()

# Calculate the sampling frequency
if len(time_diffs) > 1:
    sampling_frequency = 1 / time_diffs.mean()  # Mean accounts for irregularities
    print(f"Estimated Sampling Frequency: {sampling_frequency:.2f} Hz")
else:
    print("Not enough data to estimate sampling frequency.")
