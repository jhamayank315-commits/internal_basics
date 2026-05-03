from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

app = FastAPI()

# ---- Load model (from Task 1) ----
MODEL_PATH = "models/best_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# ---- Input Schema ----
class JobFeatures(BaseModel):
    gpu_memory_gb: int = Field(..., ge=8, le=80)
    batch_size: int = Field(..., ge=8, le=256)
    model_params_millions: int = Field(..., ge=10, le=7000)
    queue_depth: int = Field(..., ge=1, le=20)


# ---- Health Endpoint ----
@app.get("/status")
def status():
    return {
        "status": "running",
        "model": "best_model.pkl",
        "version": "1.0"
    }


# ---- Prediction Endpoint ----
@app.post("/estimate")
def estimate(features: JobFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = [[
        features.gpu_memory_gb,
        features.batch_size,
        features.model_params_millions,
        features.queue_depth
    ]]

    prediction = model.predict(data)[0]

    return {"prediction": float(prediction)}