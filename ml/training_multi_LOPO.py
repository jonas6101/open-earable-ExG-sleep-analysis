import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE
import seaborn as sns

WINDOWED_DATA_FILE = "features_combined_V3.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)
non_feature_cols = ["sleep_stage_label", "participant_id", "date", "epoch"]
feature_columns = [col for col in windowed_df.columns if col not in non_feature_cols]

label_encoder = LabelEncoder()
windowed_df["encoded_stage"] = label_encoder.fit_transform(windowed_df["sleep_stage_label"])
class_names = label_encoder.classes_  #["Awake", "Core", "Deep", "REM"]

participants = windowed_df["participant_id"].unique()
participant_results = []
confusion_matrices = []
class_specific_sensitivity = {class_name: [] for class_name in class_names}
class_specific_specificity = {class_name: [] for class_name in class_names}
overall_accuracy = []
overall_kappa = []

for participant in participants:
    #split data into training and testing
    test_df = windowed_df[windowed_df["participant_id"] == participant]
    train_df = windowed_df[windowed_df["participant_id"] != participant]

    X_train = train_df[feature_columns].values
    y_train = train_df["encoded_stage"].values
    X_test = test_df[feature_columns].values
    y_test = test_df["encoded_stage"].values

    #standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

    overall_accuracy.append(acc)
    overall_kappa.append(kappa)

    confusion_matrices.append(cm)

    #sensitivity and specificity per class
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        class_specific_sensitivity[class_name].append(sensitivity)
        class_specific_specificity[class_name].append(specificity)

    participant_results.append({
        "participant_id": participant,
        "accuracy": acc,
        "kappa": kappa,
    })


avg_accuracy = np.mean(overall_accuracy)
avg_kappa = np.mean(overall_kappa)

avg_class_sensitivity = {class_name: np.mean(sensitivities) for class_name, sensitivities in class_specific_sensitivity.items()}
avg_class_specificity = {class_name: np.mean(specificities) for class_name, specificities in class_specific_specificity.items()}

print("\nAggregate Metrics Across All Participants:")
print(f"Average Accuracy: {avg_accuracy:.3f}")
print(f"Average Cohen's Kappa: {avg_kappa:.3f}")

print("\nAverage Sensitivity per Class:")
for class_name, avg_sensitivity in avg_class_sensitivity.items():
    print(f"{class_name}: {avg_sensitivity:.3f}")

print("\nAverage Specificity per Class:")
for class_name, avg_specificity in avg_class_specificity.items():
    print(f"{class_name}: {avg_specificity:.3f}")

overall_cm = sum(confusion_matrices)
overall_cm_norm = overall_cm / overall_cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    overall_cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True, square=True,
    annot_kws={"size": 16},
    vmin=0,
    vmax=0.9
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Predicted", fontsize=20)
plt.ylabel("Actual", fontsize=20)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(participant_results)
results_df.to_csv("multistage_leaveone_results.csv", index=False)
