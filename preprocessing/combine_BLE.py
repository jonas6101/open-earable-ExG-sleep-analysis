mport pandas as pd
import os

# Specify the folder path
folder_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion4_BLE\P001\241204\rawData'

# Define the explicit header
header = ["time", "rawData1", "rawData2", "rawData3", "rawData4"]

# Create an empty list to store DataFrames
data_frames = []

# Loop over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Read the CSV file into a DataFrame assuming it already has the implicit header
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, names=header, skiprows=1)  # Add header explicitly, skipping original first row

        # Overwrite the file with the explicit header
        df.to_csv(file_path, index=False)
        print(f"Header added to {file_name}")

        # Append the DataFrame to the list
        data_frames.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined DataFrame to a new CSV file in the same folder
output_file_path = os.path.join(folder_path, 'combined_data.csv')
combined_df.to_csv(output_file_path, index=False)

print(f"Combined CSV saved at: {output_file_path}")