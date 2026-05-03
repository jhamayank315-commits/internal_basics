import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import json
import os

print('=== TASK 1: EXPERIMENT TRACKING & MODEL COMPARISON ===')

# Load data
print('Loading training data...')
data = pd.read_csv('data/training_data.csv')
X = data.drop('job_completion_min', axis=1)
y = data['job_completion_min']
print(f'Data loaded: {len(data)} rows, {len(X.columns)} features')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Data split: {len(X_train)} train, {len(X_test)} test')

# Set MLflow experiment
print('Setting up MLflow experiment: gpuforge-job-completion')
mlflow.set_experiment("gpuforge-job-completion")

models = [
    ("Ridge", Ridge()),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42))
]

results = []
best_model = None
best_rmse = float('inf')

for name, model in models:
    with mlflow.start_run(run_name=name):
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"Model: {name}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")
        
        # Log params
        if name == "Ridge":
            mlflow.log_param("alpha", model.alpha)
        elif name == "GradientBoosting":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
            mlflow.log_param("max_depth", model.max_depth)
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)
        
        # Log tag
        mlflow.set_tag("experiment_type", "baseline_comparison")
        
        # Save results dict
        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name
            # Save best model
            joblib.dump(model, 'models/best_model.pkl')

print(f"Best model: {best_model} with RMSE: {best_rmse:.2f}")

# Save results
output = {
    "experiment_name": "gpuforge-job-completion",
    "models": results,
    "best_model": best_model,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

os.makedirs('results', exist_ok=True)
with open('results/step1_tracking.json', 'w') as f:
    json.dump(output, f, indent=4)

print('\n=== TASK 1 COMPLETED ===')
print(f'Results saved to: results/step1_tracking.json')
print(f'Best model saved to: models/best_model.pkl')