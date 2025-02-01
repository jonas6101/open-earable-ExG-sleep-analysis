import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np

# Load data
WINDOWED_DATA_FILE = "features_combined_V2.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)

# Define non-feature columns
non_feature_cols = ["sleep_stage_label", "participant_id", "date", "epoch"]

# Define feature columns
feature_columns = [col for col in windowed_df.columns if col not in non_feature_cols]

# Features (X) and labels (y)
X = windowed_df[feature_columns].values
y = windowed_df["sleep_stage_label"].values

# Map sleep labels to binary classes
binary_labels = ["Awake" if label == "Awake" else "Asleep" for label in y]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(binary_labels)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Leave-One-Night-Out approach
# Select one night as the test set
test_df = windowed_df[(windowed_df['participant_id'] == 'P008')]
train_df = windowed_df.drop(test_df.index)

# Prepare training and test sets
X_train = train_df[feature_columns].values
y_train = train_df["sleep_stage_label"].values
X_test = test_df[feature_columns].values

# Map test labels to binary
binary_y_train = ["Awake" if label == "Awake" else "Asleep" for label in y_train]

# Encode binary labels
y_train_encoded = label_encoder.fit_transform(binary_y_train)

# Apply ADASYN (alternative to SMOTE) to training data
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train_encoded)

# Train Random Forest model with class weights and reduced estimators
model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
model.fit(X_train_resampled, y_train_resampled)

# Predict probabilities for test set and apply a lower threshold
probs = model.predict_proba(X_test)
threshold = 0.5
y_pred = (probs[:, 1] > threshold).astype(int)

# Encode test labels to match predicted format
binary_y_test = ["Awake" if label == "Awake" else "Asleep" for label in test_df['sleep_stage_label']]
y_test_encoded = label_encoder.transform(binary_y_test)

# Create DataFrame for each epoch with raw predictions and original labels
output_df = pd.DataFrame({
    'epoch': test_df['epoch'],
    'predicted_label': y_pred,
    'original_label': test_df['sleep_stage_label']
})

# Save predicted labels DataFrame
output_df.to_csv('predicted_epochs_with_labels.csv', index=False)

# Classification Report
print("\nClassification Report (Raw Predictions):")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# Cohen's Kappa Score
kappa_score = cohen_kappa_score(y_test_encoded, y_pred)
print(f"\nCohen's Kappa Score: {kappa_score:.2f}")
