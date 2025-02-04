import os
import pandas as pd

# This script creates feature vectors centered around one epoch for all epochs

# parameters
FEATURE_CSV_FOLDER = "../ml/features_V3"
WINDOW_BEFORE = 2
WINDOW_AFTER = 2
OUTPUT_FILE = "../ml/features_combined_V3.csv"

all_feature_csvs = [
    os.path.join(FEATURE_CSV_FOLDER, file)
    for file in os.listdir(FEATURE_CSV_FOLDER)
    if file.endswith(".csv")
]
combined_df = pd.concat([pd.read_csv(file) for file in all_feature_csvs], ignore_index=True)

combined_df["epoch"] = combined_df["epoch"].astype(int)
combined_df = combined_df.sort_values(by=["participant_id", "date", "epoch"]).reset_index(drop=True)

class_distribution = combined_df['sleep_stage_label'].value_counts(normalize=True) * 100
print(class_distribution)

feature_columns = [col for col in combined_df.columns if
                   col not in ["sleep_stage_label", "epoch", "participant_id", "date"]]

all_X_windowed = []
all_y_windowed = []
all_participant_ids = []
all_dates = []
all_center_epochs = []

# Group by participant and date
grouped = combined_df.groupby(["participant_id", "date"])

for (pid, d), group_df in grouped:
    # Extract features and labels within this single recording
    X = group_df[feature_columns].values
    y = group_df["sleep_stage_label"].values
    epochs = group_df["epoch"].values

    window_size = WINDOW_BEFORE + 1 + WINDOW_AFTER

    for i in range(WINDOW_BEFORE, len(X) - WINDOW_AFTER):
        window_epochs = X[i - WINDOW_BEFORE: i + WINDOW_AFTER + 1]
        if window_epochs.shape[0] == window_size:
            window_features = window_epochs.flatten()

            all_X_windowed.append(window_features)
            all_y_windowed.append(y[i])  # center epoch label
            all_participant_ids.append(pid)
            all_dates.append(d)
            all_center_epochs.append(epochs[i])

windowed_feature_names = []
for offset in range(-WINDOW_BEFORE, WINDOW_AFTER + 1):
    for fc in feature_columns:
        windowed_feature_names.append(f"{fc}_offset{offset}")

windowed_df = pd.DataFrame(all_X_windowed, columns=windowed_feature_names)
windowed_df["sleep_stage_label"] = all_y_windowed
windowed_df["participant_id"] = all_participant_ids
windowed_df["date"] = all_dates
windowed_df["epoch"] = all_center_epochs

windowed_df.to_csv(OUTPUT_FILE, index=False)
print(f"Windowed data saved to {OUTPUT_FILE}")
