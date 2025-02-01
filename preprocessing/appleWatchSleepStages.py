import pandas as pd

# Load filtered EEG data
filtered_file_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\P001\241126\rawData\filtered_data.csv'
eeg_df = pd.read_csv(filtered_file_path)

# First attempt to parse timestamps with fractional seconds
eeg_df['time'] = pd.to_datetime(eeg_df['time'], format='%H:%M:%S.%f', errors='coerce').dt.time

# Fill in rows where parsing failed by trying the '%H:%M:%S' format
missing_time_mask = eeg_df['time'].isna()
eeg_df.loc[missing_time_mask, 'time'] = pd.to_datetime(
    eeg_df.loc[missing_time_mask, 'time'],
    format='%H:%M:%S',
    errors='coerce'
).dt.time

# Drop rows with unparsable timestamps
eeg_df = eeg_df.dropna(subset=['time'])

# Initialize sleep stage column
eeg_df['sleep_stage'] = 'Awake'

# Path to Apple Health export XML
xml_file = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\P001\241126\apple\Export\apple_health_export\Export.xml'

# Parse XML and process as in previous code
import xml.etree.ElementTree as ET

# Define a mapping from detailed names to simpler names
stage_mapping = {
    'HKCategoryValueSleepAnalysisAsleepCore': 'Core',
    'HKCategoryValueSleepAnalysisAsleepDeep': 'Deep',
    'HKCategoryValueSleepAnalysisAsleepREM': 'REM',
    'HKCategoryValueSleepAnalysisAwake': 'Awake'
}

# Extract sleep records
records = []
tree = ET.parse(xml_file)
root = tree.getroot()

for record in root.findall('Record'):
    if record.get('type') == 'HKCategoryTypeIdentifierSleepAnalysis':
        records.append({
            'startDate': record.get('startDate'),
            'endDate': record.get('endDate'),
            'value': record.get('value')
        })

# Convert to DataFrame
sleep_df = pd.DataFrame(records)

# Convert date strings to datetime objects and remove timezone information
sleep_df['startDate'] = pd.to_datetime(sleep_df['startDate']).dt.tz_localize(None)
sleep_df['endDate'] = pd.to_datetime(sleep_df['endDate']).dt.tz_localize(None)

# Filter and map stages as before
night_start = pd.to_datetime('2024-11-26 21:00:00')
night_end = pd.to_datetime('2024-11-27 09:00:00')

sleep_df = sleep_df[(sleep_df['startDate'] >= night_start) & (sleep_df['startDate'] < night_end)]
sleep_df = sleep_df[sleep_df['value'] != 'HKCategoryValueSleepAnalysisInBed']
sleep_df['value'] = sleep_df['value'].map(stage_mapping)

sleep_df['startTime'] = sleep_df['startDate'].dt.time
sleep_df['endTime'] = sleep_df['endDate'].dt.time

# Match sleep stages to EEG data
for _, row in sleep_df.iterrows():
    mask = (eeg_df['time'] >= row['startTime']) & (eeg_df['time'] < row['endTime'])
    eeg_df.loc[mask, 'sleep_stage'] = row['value']

# Path to save combined data
combined_file_path = r'C:\Users\Jonas\OneDrive\Bachelorarbeit\open-earable-ExG-sleep-analysis\recordings\open-earable-ExG\AppVersion3_Serial\P001\241126\ml\eeg_with_sleep_stages.csv'
eeg_df.to_csv(combined_file_path, index=False)
print(f"Combined data saved at: {combined_file_path}")
