import pandas as pd

# Script to estimate the sample rate which the data was recorded with
file_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241204\rawData\eeg_with_sleep_stages.csv'
df = pd.read_csv(file_path)
try:
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f', errors='coerce')
except ValueError as e:
    print("Error parsing timestamps:", e)

df = df.dropna(subset=['time'])
time_diffs = df['time'].diff().dt.total_seconds()

if len(time_diffs) > 1:
    sampling_frequency = 1 / time_diffs.mean()  # Mean accounts for irregularities
    print(f"Estimated Sampling Frequency: {sampling_frequency:.2f} Hz")
else:
    print("Not enough data to estimate sampling frequency.")
