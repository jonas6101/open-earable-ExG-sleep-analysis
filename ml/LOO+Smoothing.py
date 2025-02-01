import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load data
WINDOWED_DATA_FILE = "features_combined_V3.csv"
windowed_df = pd.read_csv(WINDOWED_DATA_FILE)

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

# List to store prediction results
results = []

# Leave-One-Night-Out loop
nights = windowed_df['date'].unique()

for night in nights:
    # Split data into training and testing sets
    test_df = windowed_df[windowed_df['date'] == night]
    train_df = windowed_df[windowed_df['date'] != night]

    # Features and labels for training and testing
    X_train = train_df[feature_columns].values
    y_train = train_df['binary_label'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['binary_label'].values

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities and apply threshold
    probs = model.predict_proba(X_test)
    threshold = 0.4
    y_pred = (probs[:, 1] > threshold).astype(int)

    # Store results for each epoch
    for i in range(len(test_df)):
        results.append({
            "participant_id": test_df.iloc[i]["participant_id"],
            "date": test_df.iloc[i]["date"],
            "epoch": test_df.iloc[i]["epoch"],
            "true_label": y_test[i],
            "predicted_label": y_pred[i]
        })

# Create a DataFrame for results and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results_per_night.csv", index=False)

print("Results saved to 'results_per_night.csv'")
