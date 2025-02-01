import pandas as pd
import struct
from datetime import datetime, timedelta

# Sample rate and inamp_gain values
sample_rate = 256  # Replace with actual sample rate
inamp_gain = 50  # Replace with actual inamp gain if different

# Load data in chunks and write each chunk directly to the output CSV
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\NoiseTest\3'
input_file = folder_path + '/combined_data.csv'
output_file_path = folder_path + '/unpacked_data.csv'

# Open the output file in write mode and add headers
with open(output_file_path, 'w') as output_file:
    output_file.write("timestamp,raw_data\n")

    # Process CSV in chunks
    for chunk in pd.read_csv(input_file, chunksize=1000):  # Adjust chunksize as needed
        # Process each row in the chunk
        for index, row in chunk.iterrows():
            timestamp = datetime.strptime(row['time'], '%H:%M:%S.%f')
            data = struct.pack('<5f', row['rawData1'], row['rawData2'], row['rawData3'], row['rawData4'],
                               row['rawData5'])
            readings = struct.unpack('<5f', data)

            # Calculate the timestamp for each reading
            last_valid_timestamp = timestamp - timedelta(seconds=5 * 1 / sample_rate)
            for i, reading in enumerate(readings):
                # Adjust timestamp
                if i == 4:
                    timestamp_for_reading = timestamp
                else:
                    time_diff = (timestamp - last_valid_timestamp) / 5
                    timestamp_for_reading = last_valid_timestamp + (i + 1) * time_diff

                # Process only the raw data without filtering
                raw_data = (reading / inamp_gain) * 1e6

                # Write each row to the output CSV
                output_file.write(f"{timestamp_for_reading.strftime('%H:%M:%S.%f')},{raw_data}\n")

print(f"Unpacked data saved at: {output_file_path}")