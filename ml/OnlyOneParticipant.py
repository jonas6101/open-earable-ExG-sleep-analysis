import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Load data
WINDOWED_DATA_FILE = "features_combined_V2.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)

# Filter for participant P001
windowed_df = windowed_df[windowed_df['participant_id'] == 'P001']

# Define non-feature columns
non_feature_cols = ["sleep_stage_label", "participant_id", "date", "epoch"]

# Define feature columns
feature_columns = [col for col in windowed_df.columns if col not in non_feature_cols]

# Encode labels
label_encoder = LabelEncoder()
windowed_df['binary_label'] = label_encoder.fit_transform(
    ["Awake" if label == "Awake" else "Asleep" for label in windowed_df['sleep_stage_label']]
)

# Standardize features
scaler = StandardScaler()
windowed_df[feature_columns] = scaler.fit_transform(windowed_df[feature_columns])

# List to store results
combined_classification_reports = []
combined_kappa_scores = []

# Leave-One-Night-Out loop
nights = windowed_df['date'].unique()

for night in nights:
    print(f"\nResults for Night: {night}")

    # Split data into training and testing sets
    test_df = windowed_df[windowed_df['date'] == night]
    train_df = windowed_df[windowed_df['date'] != night]

    # Features and labels for training and testing
    X_train = train_df[feature_columns].values
    y_train = train_df['binary_label'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['binary_label'].values

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities and apply threshold
    probs = model.predict_proba(X_test)
    threshold = 0.5
    y_pred = (probs[:, 1] > threshold).astype(int)

    # Classification Report and Kappa Score
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    kappa = cohen_kappa_score(y_test, y_pred)

    combined_classification_reports.append(report)
    combined_kappa_scores.append(kappa)

    # Print metrics for the night
    print(f"Cohen's Kappa Score: {kappa:.2f}")
    print("Metrics for 'Asleep':")
    print(f"Precision: {report['Asleep']['precision']:.2f}, Recall: {report['Asleep']['recall']:.2f}, F1-Score: {report['Asleep']['f1-score']:.2f}")
    print("Metrics for 'Awake':")
    print(f"Precision: {report['Awake']['precision']:.2f}, Recall: {report['Awake']['recall']:.2f}, F1-Score: {report['Awake']['f1-score']:.2f}")

# Combine results
average_kappa = np.mean(combined_kappa_scores)
print(f"\nAverage Cohen's Kappa Score: {average_kappa:.2f}")

# Aggregate classification metrics
avg_precision_asleep = np.mean([report['Asleep']['precision'] for report in combined_classification_reports])
avg_recall_asleep = np.mean([report['Asleep']['recall'] for report in combined_classification_reports])
avg_f1_asleep = np.mean([report['Asleep']['f1-score'] for report in combined_classification_reports])

avg_precision_awake = np.mean([report['Awake']['precision'] for report in combined_classification_reports])
avg_recall_awake = np.mean([report['Awake']['recall'] for report in combined_classification_reports])
avg_f1_awake = np.mean([report['Awake']['f1-score'] for report in combined_classification_reports])

print("\nAverage Metrics for 'Asleep':")
print(f"Precision: {avg_precision_asleep:.2f}, Recall: {avg_recall_asleep:.2f}, F1-Score: {avg_f1_asleep:.2f}")

print("\nAverage Metrics for 'Awake':")
print(f"Precision: {avg_precision_awake:.2f}, Recall: {avg_recall_awake:.2f}, F1-Score: {avg_f1_awake:.2f}")
