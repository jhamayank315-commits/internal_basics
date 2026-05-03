import json
import pandas as pd
import os

print('=== TASK 3: PREDICTION LOGGING & MONITORING ===')

# Load training data
print('Loading training data...')
data = pd.read_csv('data/training_data.csv')
train_mean_params = data['model_params_millions'].mean()
train_mean_queue = data['queue_depth'].mean()
print(f'Training data: params_mean={train_mean_params:.2f}, queue_mean={train_mean_queue:.2f}')

# Load predictions
predictions = []
with open('logs/predictions.jsonl', 'r') as f:
    for line in f:
        predictions.append(json.loads(line))

live_params = [p['input']['model_params_millions'] for p in predictions]
live_queue = [p['input']['queue_depth'] for p in predictions]
live_mean_params = sum(live_params) / len(live_params)
live_mean_queue = sum(live_queue) / len(live_queue)

shift_params = live_mean_params - train_mean_params
shift_queue = live_mean_queue - train_mean_queue

status_params = "ALERT" if shift_params > 500 else "OK"
status_queue = "ALERT" if shift_queue > 5 else "OK"

print(f"model_params_millions: train_mean={train_mean_params:.2f}, live_mean={live_mean_params:.2f}, shift={shift_params:.2f}, threshold=500, status={status_params}")
print(f"queue_depth: train_mean={train_mean_queue:.2f}, live_mean={live_mean_queue:.2f}, shift={shift_queue:.2f}, threshold=5, status={status_queue}")

# For json
mean_pred = sum(p['prediction'] for p in predictions) / len(predictions)
drift_detected = status_params == "ALERT" or status_queue == "ALERT"

alerts = [
    {
        "feature": "model_params_millions",
        "train_mean": train_mean_params,
        "live_mean": live_mean_params,
        "shift": shift_params,
        "threshold": 500,
        "status": status_params
    },
    {
        "feature": "queue_depth",
        "train_mean": train_mean_queue,
        "live_mean": live_mean_queue,
        "shift": shift_queue,
        "threshold": 5,
        "status": status_queue
    }
]

output = {
    "total_predictions": len(predictions),
    "mean_prediction": mean_pred,
    "drift_detected": drift_detected,
    "alerts": alerts
}

os.makedirs('results', exist_ok=True)
with open('results/step3_monitoring.json', 'w') as f:
    json.dump(output, f, indent=4)

print(f'\nTotal predictions: {len(predictions)}')
print(f'Mean prediction: {mean_pred:.2f}')
print(f'Drift detected: {drift_detected}')
print('Alerts:')
for alert in alerts:
    print(f"  - {alert['feature']}: {alert['status']}")
print('\n=== TASK 3 COMPLETED ===')
print(f'Results saved to: results/step3_monitoring.json')