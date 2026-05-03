import requests
import json
import os

BASE_URL = "http://localhost:8500"

# Required test input
test_input = {
    "gpu_memory_gb": 40,
    "batch_size": 64,
    "model_params_millions": 1500,
    "queue_depth": 8
}

# Call endpoints
health = requests.get(f"{BASE_URL}/status").json()
prediction = requests.post(f"{BASE_URL}/estimate", json=test_input).json()

# Prepare output
output = {
    "health_endpoint": "/status",
    "predict_endpoint": "/estimate",
    "port": 8500,
    "health_response": health,
    "test_input": test_input,
    "prediction": float(prediction.get("prediction", 0.0))
}

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Save file
with open("results/step2_serving.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ File created: results/step2_serving.json")
print(json.dumps(output, indent=4))