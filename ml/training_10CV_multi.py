import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

WINDOWED_DATA_FILE = "features_combined_V3.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)

non_feature_cols = ["sleep_stage_label", "participant_id", "date", "epoch"]
feature_columns = [col for col in windowed_df.columns if col not in non_feature_cols]

X = windowed_df[feature_columns].values
y = windowed_df["sleep_stage_label"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10-fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

# Initialize metrics
accuracy_scores = []
kappa_scores = []
conf_matrices = []
class_specific_sensitivity = {class_name: [] for class_name in label_encoder.classes_}
class_specific_specificity = {class_name: [] for class_name in label_encoder.classes_}

for train_index, test_index in kf.split(X_scaled, y_encoded):
    # Split train and test
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    kappa = cohen_kappa_score(y_test, y_pred)
    kappa_scores.append(kappa)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    conf_matrices.append(cm)

    # sensitivity and specificity per classl
    for i, class_name in enumerate(label_encoder.classes_):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        class_specific_sensitivity[class_name].append(sensitivity)
        class_specific_specificity[class_name].append(specificity)

mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_kappa = np.mean(kappa_scores)

# Average confusion matrix
avg_conf_mat = np.mean(conf_matrices, axis=0)

# Normalize the confusion matrix
avg_conf_mat_normalized = avg_conf_mat / avg_conf_mat.sum(axis=1, keepdims=True)

avg_class_sensitivity = {class_name: np.mean(sensitivities) for class_name, sensitivities in class_specific_sensitivity.items()}
avg_class_specificity = {class_name: np.mean(specificities) for class_name, specificities in class_specific_specificity.items()}

print("\n10-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Accuracy Standard Deviation: {std_accuracy:.2f}")
print(f"Mean Cohen's Kappa: {mean_kappa:.2f}")

print("\nAverage Sensitivity per Class:")
for class_name, avg_sensitivity in avg_class_sensitivity.items():
    print(f"{class_name}: {avg_sensitivity:.3f}")

print("\nAverage Specificity per Class:")
for class_name, avg_specificity in avg_class_specificity.items():
    print(f"{class_name}: {avg_specificity:.3f}")

plt.figure(figsize=(10, 8))
sns.heatmap(avg_conf_mat_normalized, annot=True, fmt='.2f', cmap='Blues',  cbar=True, square=True, vmin=0, vmax=0.9,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 16})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Predicted", fontsize=20)
plt.ylabel("Actual", fontsize=20)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)
plt.tight_layout()
plt.show()
