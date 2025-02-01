from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd

# Load data
DATA_FILE = "../ml/features_combined_V2.csv"
df = pd.read_csv(DATA_FILE)

# Define features and labels
non_feature_cols = ['sleep_stage_label', 'epoch', 'participant_id', 'date']
feature_cols = [col for col in df.columns if col not in non_feature_cols]
X = df[feature_cols]
y = df['sleep_stage_label']

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None]  # Include class weight
}

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter combinations to try
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Perform the search
random_search.fit(X, y)

# Best parameters
best_rf_params = random_search.best_params_
print("Best Parameters:", best_rf_params)
