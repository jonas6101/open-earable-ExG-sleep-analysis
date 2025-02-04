from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd
import seaborn as sns
import numpy as np

WINDOWED_DATA_FILE = "features_combined_V3.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)

non_feature_cols = ["sleep_stage_label", "participant_id", "date", "epoch"]
feature_columns = [col for col in windowed_df.columns if col not in non_feature_cols]
windowed_df['binary_label'] = windowed_df['sleep_stage_label'].apply(
    lambda label: 1 if label == "Awake" else 0
)

# Leave-All-Recordings-from-One-Participant-Out
participants = windowed_df['participant_id'].unique()
participant_results = []
confusion_matrices = []

# initialize accumulators for metrics
overall_accuracy = []
overall_kappa = []
sensitivity_asleep = []
specificity_asleep = []
sensitivity_awake = []
specificity_awake = []

for participant in participants:
    # sdplit data into training and testing sets
    test_df = windowed_df[windowed_df['participant_id'] == participant]
    train_df = windowed_df[windowed_df['participant_id'] != participant]
    X_train = train_df[feature_columns].values
    y_train = train_df['binary_label'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['binary_label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train_resampled, y_train_resampled)

    # predict probabilities and classify based on threshold. 0.5 is the standard in the RandomForestClassifier itself
    probs = model.predict_proba(X_test_scaled)
    threshold = 0.5
    y_pred = (probs[:, 1] > threshold).astype(int)

    # calulate metrics per class, per participant and overall
    acc = np.mean(y_pred == y_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_matrices.append(cm)


    TP_awake = cm[1, 1]
    FP_awake = cm[0, 1]
    FN_awake = cm[1, 0]
    TN_awake = cm[0, 0]
    sensitivity_awake_value = TP_awake / (TP_awake + FN_awake) if TP_awake + FN_awake > 0 else 0
    specificity_awake_value = TN_awake / (TN_awake + FP_awake) if TN_awake + FP_awake > 0 else 0

    TP_asleep = cm[0, 0]
    FP_asleep = cm[1, 0]
    FN_asleep = cm[0, 1]
    TN_asleep = cm[1, 1]
    sensitivity_asleep_value = TP_asleep / (TP_asleep + FN_asleep) if TP_asleep + FN_asleep > 0 else 0
    specificity_asleep_value = TN_asleep / (TN_asleep + FP_asleep) if TN_asleep + FP_asleep > 0 else 0

    sensitivity_asleep.append(sensitivity_asleep_value)
    specificity_asleep.append(specificity_asleep_value)
    sensitivity_awake.append(sensitivity_awake_value)
    specificity_awake.append(specificity_awake_value)
    overall_accuracy.append(acc)
    overall_kappa.append(kappa)

    participant_results.append({
        "participant_id": participant,
        "accuracy": acc,
        "kappa": kappa,
        "sensitivity_awake": sensitivity_awake_value,
        "specificity_awake": specificity_awake_value
    })

average_cm = np.mean(confusion_matrices, axis=0)

avg_accuracy = np.mean(overall_accuracy)
avg_kappa = np.mean(overall_kappa)
avg_sensitivity_asleep = np.mean(sensitivity_asleep)
avg_specificity_asleep = np.mean(specificity_asleep)
avg_sensitivity_awake = np.mean(sensitivity_awake)
avg_specificity_awake = np.mean(specificity_awake)

print("\nAggregate Metrics Across All Participants:")
print(f"Average Accuracy: {avg_accuracy:.3f}")
print(f"Average Cohen's Kappa: {avg_kappa:.3f}")

print("\nAverage Metrics for Asleep Class:")
print(f"Sensitivity: {avg_sensitivity_asleep:.3f}, Specificity: {avg_specificity_asleep:.3f}")

print("\nAverage Metrics for Awake Class:")
print(f"Sensitivity: {avg_sensitivity_awake:.3f}, Specificity: {avg_specificity_awake:.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(average_cm / average_cm.sum(axis=1, keepdims=True), annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
            xticklabels=["Asleep", "Awake"], yticklabels=["Asleep", "Awake"], cbar=True, square=True, annot_kws={"size": 20})

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Predicted", fontsize=20)
plt.ylabel("Actual", fontsize=20)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)

plt.savefig(f"average_confusion_matrix_binary.png", dpi=300, bbox_inches='tight')
plt.show()
results_df = pd.DataFrame(participant_results)
results_df.to_csv("binary_classification_results.csv", index=False)
