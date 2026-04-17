"""
FastAPI Inference Service for ARES.
Loads the baseline XGBoost model from the local MLflow tracking registry and exposes a POST /predict endpoint.
Logs inference requests to a local SQLite database for drift monitoring.
"""

import os
import sqlite3
import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

app = FastAPI(title="ARES Inference Service", version="1.0.0")

# --- Initialize SQLite Database ---
DB_PATH = "data/inference_logs.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# Enable WAL mode for better concurrency
conn.execute('PRAGMA journal_mode=WAL;')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS inference_logs
                (timestamp TEXT, user_id REAL, product_id REAL, price REAL, 
                 event_type TEXT, is_high_value BOOLEAN, fraud_probability REAL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS retraining_jobs
                (job_id TEXT PRIMARY KEY, 
                 start_time TEXT, 
                 status TEXT, 
                 champion_f1 REAL, 
                 challenger_f1 REAL, 
                 decision TEXT, 
                 shap_path TEXT,
                 psi_score REAL)''')
conn.commit()

# --- Load Model from MLflow ---
model = None
try:
    print("Fetching the latest model from ARES_Phase1_Baseline experiment...")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ARES_Phase1_Baseline")
    if experiment is None:
        raise ValueError("Could not find MLflow experiment 'ARES_Phase1_Baseline'")
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
                              order_by=["start_time DESC"], max_results=10)
    if not runs:
        raise ValueError("No runs found for experiment.")
        
    for run in runs:
        if run.info.status != "FINISHED":
            continue
            
        latest_run_id = run.info.run_id
        for path in ["xgboost_baseline_model", "model"]:
            try:
                model_uri = f"runs:/{latest_run_id}/{path}"
                model = mlflow.xgboost.load_model(model_uri)
                print(f"Successfully loaded XGBoost model from run: {latest_run_id}")
                break
            except Exception:
                continue
        if model is not None:
            break
            
    if model is None:
        raise ValueError("Could not find any valid model artifacts in recent runs.")
except Exception as e:
    print(f"Warning: Failed to load model from MLflow. Ensure 'python src/baseline_trainer.py' was run. Error: {e}")

# --- API Data Models ---
class InferenceRequest(BaseModel):
    user_id: float
    product_id: float
    price: float
    event_type: str
    is_high_value: bool

class InferenceResponse(BaseModel):
    fraud_probability: float

# Event Type Encoder (mimicking LabelEncoder mapping from baseline)
# In standard baseline, categorical string ['cart', 'purchase', 'view'] orders natively alphabetically: cart=0, purchase=1, view=2
EVENT_MAP = {'cart': 0, 'purchase': 1, 'view': 2}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if model is None:
         raise HTTPException(status_code=503, detail="Model is not loaded.")
         
    # 1. Feature Preprocessing
    encoded_event = EVENT_MAP.get(request.event_type.lower(), 2) # Default to 'view'
    
    features = pd.DataFrame([{
        'user_id': request.user_id,
        'event_type': encoded_event,
        'product_id': request.product_id,
        'price': request.price
    }])
    
    # 2. Prediction
    # extract probability for the positive class (1: fraud)
    prob_fraud = float(model.predict_proba(features)[0][1])
    
    # 3. Logging to SQLite for Drift Detection
    timestamp = datetime.datetime.now().isoformat()
    try:
        cursor.execute('''INSERT INTO inference_logs 
                          VALUES (?, ?, ?, ?, ?, ?, ?)''',
                       (timestamp, request.user_id, request.product_id, request.price, 
                        request.event_type, request.is_high_value, prob_fraud))
        conn.commit()
    except Exception as e:
        print(f"Error logging to SQLite: {e}")
        
    return InferenceResponse(fraud_probability=prob_fraud)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
