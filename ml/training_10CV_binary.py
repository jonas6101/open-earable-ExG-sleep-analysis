import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# Load data
WINDOWED_DATA_FILE = "features_combined_V3_no_windowing.csv"
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

# 10-fold cross-validation setup
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Custom threshold for probability
threshold = 0.5

# Lists to aggregate predictions/labels from all folds
all_y_test = []
all_y_pred = []
accuracy_scores = []
kappa_scores = []
sensitivity_asleep = []
specificity_asleep = []
sensitivity_awake = []
specificity_awake = []

for train_index, test_index in kf.split(X_scaled, y_encoded):
    # Train/Test split
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Apply SMOTE oversampling only on the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train model on resampled data
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities instead of labels
    probs = model.predict_proba(X_test)

    # Apply threshold for binary classification
    y_pred = (probs[:, 1] > threshold).astype(int)

    # Collect for aggregated confusion matrix
    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

    # Metrics for this fold
    acc = np.mean(y_pred == y_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy_scores.append(acc)
    kappa_scores.append(kappa)

    # Confusion matrix for this fold
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    TP_awake = cm[1, 1]
    FP_awake = cm[0, 1]
    FN_awake = cm[1, 0]
    TN_awake = cm[0, 0]

    sensitivity_awake.append(TP_awake / (TP_awake + FN_awake) if TP_awake + FN_awake > 0 else 0)
    specificity_awake.append(TN_awake / (TN_awake + FP_awake) if TN_awake + FP_awake > 0 else 0)

    TP_asleep = cm[0, 0]
    FP_asleep = cm[1, 0]
    FN_asleep = cm[0, 1]
    TN_asleep = cm[1, 1]

    sensitivity_asleep.append(TP_asleep / (TP_asleep + FN_asleep) if TP_asleep + FN_asleep > 0 else 0)
    specificity_asleep.append(TN_asleep / (TN_asleep + FP_asleep) if TN_asleep + FP_asleep > 0 else 0)

# Compute aggregated confusion matrix
cm_aggregated = confusion_matrix(all_y_test, all_y_pred, labels=[0, 1])
cm_normalized = cm_aggregated / cm_aggregated.sum(axis=1, keepdims=True)

# Compute average metrics
avg_accuracy = np.mean(accuracy_scores)
avg_kappa = np.mean(kappa_scores)
avg_sensitivity_asleep = np.mean(sensitivity_asleep)
avg_specificity_asleep = np.mean(specificity_asleep)
avg_sensitivity_awake = np.mean(sensitivity_awake)
avg_specificity_awake = np.mean(specificity_awake)

# Print aggregate metrics
print("\n10-Fold Cross-Validation Aggregate Metrics:")
print(f"Average Accuracy: {avg_accuracy:.3f}")
print(f"Average Cohen's Kappa: {avg_kappa:.3f}")

print("\nAverage Metrics for Asleep Class:")
print(f"Sensitivity: {avg_sensitivity_asleep:.3f}, Specificity: {avg_specificity_asleep:.3f}")

print("\nAverage Metrics for Awake Class:")
print(f"Sensitivity: {avg_sensitivity_awake:.3f}, Specificity: {avg_specificity_awake:.3f}")

#normalized confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,  cbar=True, square=True,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 20})
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Predicted", fontsize=20)
plt.ylabel("Actual", fontsize=20)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)


plt.savefig("confusion_matrix_binary_classification.png", dpi=300, bbox_inches='tight')

plt.show()

# classification Report
print("\nClassification Report:")
print(classification_report(all_y_test, all_y_pred, target_names=label_encoder.classes_))
