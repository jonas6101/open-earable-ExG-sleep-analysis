import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

input_csv = r'../recordings/open-earable-ExG/AppVersion4_BLE/P009/250118/rawData/eeg_with_sleep_stages.csv'  # Replace with your input CSV path
output_folder = r'../recordings/open-earable-ExG/AppVersion4_BLE/P009/250118/epochs'  # Replace with your desired output folder path

# parameters
epoch_duration = 30
cutoff_time = "12:30:00"  #cutoff as time of day
value_threshold = 500  #threshold for exclusion (±500 µV)

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_csv)

if df.empty:
    raise ValueError("The input CSV is empty.")


def parse_time(row_time):
    try:

        return datetime.strptime(row_time, '%H:%M:%S.%f').time()
    except ValueError:
        try:
            return datetime.strptime(row_time, '%H:%M:%S').time()
        except ValueError:
            return None


df['time'] = df['time'].apply(parse_time)


if df['time'].isnull().any():
    print("Unparsed rows detected. Saving problematic rows for review.")
    problematic_rows = df[df['time'].isnull()]
    problematic_rows.to_csv('unparsed_times.csv', index=False)
    raise ValueError("Some rows in the 'time' column could not be parsed. Check 'unparsed_times.csv' for details.")

cutoff_time = datetime.strptime(cutoff_time, '%H:%M:%S').time()

df = df[df['time'] < cutoff_time]

if df.empty:
    raise ValueError("No data remains after applying the cutoff time filter.")

start_time = df['time'].iloc[0]
end_time = df['time'].iloc[-1]

print(f"Start time: {start_time}, End time: {end_time}")

def add_seconds_to_time(t, seconds):
    dt = datetime.combine(datetime.today(), t) + timedelta(seconds=seconds)
    return dt.time()

current_time = start_time
epoch_index = 0

while current_time < end_time:
    epoch_start = current_time
    epoch_end = add_seconds_to_time(epoch_start, epoch_duration)

    epoch_data = df[(df['time'] >= epoch_start) & (df['time'] < epoch_end)]

    if epoch_data.empty:
        break

    max_value = epoch_data['filtered_data'].max()
    min_value = epoch_data['filtered_data'].min()

    if max_value > value_threshold or min_value < -value_threshold:
        print(f"Excluding epoch {epoch_index}: max = {max_value}, min = {min_value}")
    else:
        epoch_file_path = os.path.join(output_folder, f"epoch_{epoch_index:04d}.csv")
        epoch_data.to_csv(epoch_file_path, index=False)
        print(f"Saved clean epoch {epoch_index} to {epoch_file_path}")

    # Move to the next epoch
    current_time = epoch_end
    epoch_index += 1

print(f"Epoch splitting completed. Clean epochs saved in: {output_folder}")
