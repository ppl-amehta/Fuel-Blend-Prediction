import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import joblib
import os

# --- Configuration ---
# Create a directory to save the trained models
os.makedirs('models', exist_ok=True)

# --- Load Data ---
try:
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# --- Feature Engineering ---
# Create interaction features as before
components = [f'Component{i}' for i in range(1, 6)]
properties = [f'Property{i}' for i in range(1, 11)]
blend_properties = [f'BlendProperty{i}' for i in range(1, 11)]

for df in [train_df, test_df]:
    for component in components:
        for prop in properties:
            # Interaction term: component fraction * component property
            df[f'{component}_{prop}_interaction'] = df[f'{component}_fraction'] * df[f'{component}_{prop}']

# Define the features to be used for training
features = [col for col in train_df.columns if col.startswith('Component')]
X = train_df[features]
X_test = test_df[features]

# --- Hyperparameter Tuning and Model Training ---
def objective(trial, X, y):
    """
    Objective function for Optuna to minimize.
    This function trains a LightGBM model with a set of hyperparameters
    suggested by Optuna and returns the cross-validated MAPE.
    """
    # Define the hyperparameter search space for LightGBM
    param = {
        'objective': 'regression_l1',
        'metric': 'mape',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
    }

    # Use K-Fold cross-validation to get a robust estimate of the model's performance
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mape_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        preds = model.predict(X_val)
        mape_scores.append(mean_absolute_percentage_error(y_val, preds))

    return np.mean(mape_scores)

# Dictionary to store the best models
best_models = {}

# Iterate over each target property to tune and train a model
for target in blend_properties:
    print(f"--- Tuning and Training for {target} ---")
    y = train_df[target]

    # Create an Optuna study to find the best hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50) # 50 trials for a good balance of speed and performance

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best MAPE for {target}: {study.best_value}")
    print(f"Best hyperparameters for {target}: {best_params}")

    # Train the final model with the best hyperparameters on the entire training set
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(final_model, f'models/{target}_model.joblib')
    print(f"Saved best model for {target}")

    best_models[target] = final_model

# --- Prediction ---
predictions = {}
for target in blend_properties:
    print(f"Predicting {target}...")
    model = best_models[target]
    predictions[target] = model.predict(X_test)

# --- Create Submission File ---
submission_df = pd.DataFrame({'ID': test_df['ID']})
for target in blend_properties:
    submission_df[target] = predictions[target]

submission_df.to_csv('submission_v2.csv', index=False)
print("\nSubmission file 'submission_v2.csv' created successfully!")