import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder

# Load the full dataset
DATA_FILE = "../ml/features_combined_V2.csv"
df = pd.read_csv(DATA_FILE)

# Define features and labels
non_feature_cols = ['sleep_stage_label', 'epoch', 'participant_id', 'date']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
X = df[feature_cols]
y = df['sleep_stage_label']

# Encode the labels
y_encoded = LabelEncoder().fit_transform(y)

# Initialize the model and RFE
model = RandomForestClassifier(random_state=42, n_jobs=-1)
n_features_to_select = 50  # Number of features to keep
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=1)

# Fit RFE
rfe.fit(X, y_encoded)

# Get the selected features
selected_features = [feature for feature, selected in zip(feature_cols, rfe.support_) if selected]

# Save the selected features and their rankings
feature_ranking = pd.DataFrame({
    'Feature': feature_cols,
    'Ranking': rfe.ranking_
}).sort_values(by='Ranking')

# Print and save the results
print(f"Selected Features ({n_features_to_select}):")
print(selected_features)
feature_ranking.to_csv('rfe_selected_features.csv', index=False)
