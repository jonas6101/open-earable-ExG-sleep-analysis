#This script calculates the correct timestamps for the input csv.
# Reason for that is that the timestamp generation on the arduino does not match the system time of the phone or the apple watch

import pandas as pd
from datetime import datetime, timedelta

file_path = r"C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241201\rawData\unpacked_data1.csv"
df = pd.read_csv(file_path)

# define the real start time and the Arduino's first timestamp
real_start_time = datetime.strptime("08:00:00", "%H:%M:%S")
arduino_start_time = datetime.strptime(df.iloc[0]['timestamp'], "%H:%M:%S.%f")

offset = real_start_time - arduino_start_time

df['timestamp'] = df['timestamp'].apply(
    lambda x: (datetime.strptime(x, "%H:%M:%S.%f") + offset).strftime("%H:%M:%S.%f")
)

output_path = r"C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241201\rawData\adjusted_timestamps.csv"
df.to_csv(output_path, index=False)
