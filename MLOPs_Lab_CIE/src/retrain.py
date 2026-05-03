import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import json
import os

print('=== TASK 4: RETRAINING PIPELINE ===')

# Load data
print('Loading data...')
data = pd.read_csv('data/training_data.csv')
new_data = pd.read_csv('data/new_data.csv')
print(f'Original data: {len(data)} rows')
print(f'New data: {len(new_data)} rows')

combined = pd.concat([data, new_data])
print(f'Combined data: {len(combined)} rows')

X_combined = combined.drop('job_completion_min', axis=1)
y_combined = combined['job_completion_min']

# The same test set as Task 1
X = data.drop('job_completion_min', axis=1)
y = data['job_completion_min']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Champion rmse from Task 1
champion_rmse = 2.74

# Retrain
retrained = GradientBoostingRegressor(random_state=42)
retrained.fit(X_combined, y_combined)

y_pred_retrained = retrained.predict(X_test)
retrained_rmse = np.sqrt(mean_squared_error(y_test, y_pred_retrained))

improvement = champion_rmse - retrained_rmse

min_improvement_threshold = 0.5

if improvement > min_improvement_threshold:
    action = "promoted"
else:
    action = "kept_champion"

output = {
    "original_data_rows": len(data),
    "new_data_rows": len(new_data),
    "combined_data_rows": len(combined),
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": min_improvement_threshold,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs('results', exist_ok=True)
with open('results/step4_retraining.json', 'w') as f:
    json.dump(output, f, indent=4)

print(f'\nChampion RMSE: {champion_rmse:.2f}')
print(f'Retrained RMSE: {retrained_rmse:.2f}')
print(f'Improvement: {improvement:.2f}')
print(f'Minimum threshold: {min_improvement_threshold:.2f}')
print(f'Action: {action.upper()}')
print('\n=== TASK 4 COMPLETED ===')
print(f'Results saved to: results/step4_retraining.json')