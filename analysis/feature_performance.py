import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
DATA_FILE = "../ml/features_combined_V2.csv"
df = pd.read_csv(DATA_FILE)

# Define features and labels
non_feature_cols = ['sleep_stage_label', 'epoch', 'participant_id', 'date']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
X = df[feature_cols]
y = df['sleep_stage_label']

# Encode labels
y_encoded = LabelEncoder().fit_transform(y)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

# Get feature importances
importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})

# Combine importance by feature name without offsets
feature_importance['BaseFeature'] = feature_importance['Feature'].str.extract(r'^(.*?)(?:_offset[+-]?\d+)?$')
aggregated_importance = feature_importance.groupby('BaseFeature')['Importance'].sum().reset_index()
aggregated_importance = aggregated_importance.sort_values(by='Importance', ascending=False)

# Print aggregated importances
print("Aggregated Feature Importances:")
print(aggregated_importance)

# Save aggregated importances to CSV
aggregated_importance.to_csv('aggregated_feature_importances.csv', index=False)
