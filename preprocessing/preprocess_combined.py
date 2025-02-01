import os
import pandas as pd
import numpy as np
import struct
from datetime import datetime, timedelta
import scipy.signal
import xml.etree.ElementTree as ET

# Define paths and parameters
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P009\250118\rawData'
final_output_path = os.path.join(folder_path, 'eeg_with_sleep_stages.csv')
xml_file_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P009\250118\appleHealth\Export\apple_health_export\Export.xml'

# Parameters
start_time_of_recording = "04:43:52.000000"
night_start = pd.to_datetime('2025-01-18 21:00:00')
night_end = pd.to_datetime('2025-01-19 12:00:00')
sample_rate = 241
inamp_gain = 50

# Create filters
sos_bandpass = scipy.signal.iirfilter(4, Wn=[0.5, 32], fs=sample_rate, btype="bandpass", ftype="butter", output="sos")

# Step 1: Combine CSV files
header = ["time", "rawData1", "rawData2", "rawData3", "rawData4"]
data_frames = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df_temp = pd.read_csv(file_path, names=header, skiprows=1)
        data_frames.append(df_temp)

# Combine all data into a single DataFrame
df = pd.concat(data_frames, ignore_index=True)
print("Data combined.")

# Step 2: Unpack and scale raw data
unpacked_data = []
for _, row in df.iterrows():
    time_str = row['time']

    # Try parsing with microseconds first
    try:
        timestamp = datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        # If that fails, try without microseconds
        timestamp = datetime.strptime(time_str, '%H:%M:%S')

    data = struct.pack('<4f', row['rawData1'], row['rawData2'], row['rawData3'], row['rawData4'])
    readings = struct.unpack('<4f', data)

    # Calculate timestamps for each of the 4 readings
    last_valid_timestamp = timestamp - timedelta(seconds=4 * 1 / sample_rate)
    for i, reading in enumerate(readings):
        timestamp_for_reading = (
            timestamp if i == 3
            else last_valid_timestamp + (i + 1) * (timestamp - last_valid_timestamp) / 4
        )
        raw_data = (reading / inamp_gain) * 1e6
        unpacked_data.append({'timestamp': timestamp_for_reading, 'raw_data': raw_data})

df = pd.DataFrame(unpacked_data)
print("Data unpacked.")

# Step 3: Adjust timestamps
real_start_time = datetime.strptime(start_time_of_recording, '%H:%M:%S.%f')
arduino_start_time = df['timestamp'].iloc[0]
offset = real_start_time - arduino_start_time
df['timestamp'] = df['timestamp'] + offset
print("Timestamps adjusted.")

# Step 4: Apply bandpass and notch filters
def apply_notch_filter(data, freq, sample_rate, quality_factor=5):
    b_notch, a_notch = scipy.signal.iirnotch(freq, quality_factor, fs=sample_rate)
    return scipy.signal.filtfilt(b_notch, a_notch, data)

# Apply notch filters for 50Hz, 100Hz
y_filtered = apply_notch_filter(df['raw_data'], 50, sample_rate)
y_filtered = apply_notch_filter(y_filtered, 100, sample_rate)

# Bandpass filtering
df['filtered_data'] = scipy.signal.sosfiltfilt(sos_bandpass, y_filtered)
print("Data filtered.")

# Step 5: Derive 'time' column and integrate sleep stages
df['time'] = pd.to_datetime(df['timestamp']).dt.time
df.drop(columns=['timestamp'], inplace=True)
df['sleep_stage'] = 'Awake'

# Mapping sleep stages from XML
stage_mapping = {
    'HKCategoryValueSleepAnalysisAsleepCore': 'Core',
    'HKCategoryValueSleepAnalysisAsleepDeep': 'Deep',
    'HKCategoryValueSleepAnalysisAsleepREM': 'REM',
    'HKCategoryValueSleepAnalysisAwake': 'Awake'
}

records = []
tree = ET.parse(xml_file_path)
root = tree.getroot()
for record in root.findall('Record'):
    if record.get('type') == 'HKCategoryTypeIdentifierSleepAnalysis':
        records.append({
            'startDate': record.get('startDate'),
            'endDate': record.get('endDate'),
            'value': record.get('value')
        })

sleep_df = pd.DataFrame(records)
sleep_df['startDate'] = pd.to_datetime(sleep_df['startDate']).dt.tz_localize(None)
sleep_df['endDate'] = pd.to_datetime(sleep_df['endDate']).dt.tz_localize(None)
sleep_df = sleep_df[(sleep_df['startDate'] >= night_start) & (sleep_df['startDate'] < night_end)]
sleep_df = sleep_df[sleep_df['value'] != 'HKCategoryValueSleepAnalysisInBed']
sleep_df['value'] = sleep_df['value'].map(stage_mapping)
sleep_df['startTime'] = sleep_df['startDate'].dt.time
sleep_df['endTime'] = sleep_df['endDate'].dt.time

# Integrate sleep stages into df
for _, row in sleep_df.iterrows():
    mask = (df['time'] >= row['startTime']) & (df['time'] < row['endTime'])
    df.loc[mask, 'sleep_stage'] = row['value']

# Save the final DataFrame
df.to_csv(final_output_path, index=False)
print(f"Final preprocessed data saved at: {final_output_path}")
