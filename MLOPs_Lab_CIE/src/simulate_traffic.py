import requests
import pandas as pd
import random

print('=== TASK 3: SIMULATING TRAFFIC ===')
print('Loading data...')

data = pd.read_csv('data/training_data.csv')
new_data = pd.read_csv('data/new_data.csv')

url = "http://localhost:8500/estimate"

print('Sending 40 normal requests...')
# 40 normal
for i in range(40):
    row = data.iloc[random.randint(0, 24)]
    payload = {
        "gpu_memory_gb": int(row['gpu_memory_gb']),
        "batch_size": int(row['batch_size']),
        "model_params_millions": float(row['model_params_millions']),
        "queue_depth": int(row['queue_depth'])
    }
    try:
        requests.post(url, json=payload, timeout=5)
        if (i + 1) % 10 == 0:
            print(f"  Sent {i + 1} normal requests")
    except Exception as e:
        print(f"  Error: {e}")

print('Sending 10 drifted requests...')
# 10 drifted
for i in range(10):
    row = new_data.iloc[random.randint(0, 19)]
    payload = {
        "gpu_memory_gb": int(row['gpu_memory_gb']),
        "batch_size": int(row['batch_size']),
        "model_params_millions": float(row['model_params_millions']),
        "queue_depth": int(row['queue_depth'])
    }
    try:
        requests.post(url, json=payload, timeout=5)
        if (i + 1) % 5 == 0:
            print(f"  Sent {i + 1} drifted requests")
    except Exception as e:
        print(f"  Error: {e}")

print('\n=== TRAFFIC SIMULATION COMPLETED ===')
print(f'Total requests sent: 50 (40 normal + 10 drifted)')