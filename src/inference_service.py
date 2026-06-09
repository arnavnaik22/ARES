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

try:
    from src.feature_schema import encode_feature_frame
except ImportError:
    from feature_schema import encode_feature_frame

app = FastAPI(title="ARES Inference Service", version="1.0.0")

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

# --- Initialize SQLite Database ---
DB_PATH = "data/inference_logs.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# Enable WAL mode for better concurrency
conn.execute('PRAGMA journal_mode=WAL;')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS inference_logs
                (timestamp TEXT, user_id REAL, product_id REAL, price REAL, 
                 event_type TEXT, category TEXT, device_type TEXT, channel TEXT, country TEXT,
                 session_duration REAL, account_age_days REAL, prior_orders REAL, prior_chargebacks REAL,
                 discount_pct REAL, shipping_speed TEXT, hour_of_day REAL, is_weekend INTEGER, cart_size REAL,
                 merchant_risk_score REAL, is_high_value BOOLEAN, is_fraud BOOLEAN, fraud_probability REAL)''')

def ensure_inference_schema():
    cursor.execute("PRAGMA table_info(inference_logs)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    desired_columns = {
        "timestamp": "TEXT",
        "user_id": "REAL",
        "product_id": "REAL",
        "price": "REAL",
        "event_type": "TEXT",
        "category": "TEXT",
        "device_type": "TEXT",
        "channel": "TEXT",
        "country": "TEXT",
        "session_duration": "REAL",
        "account_age_days": "REAL",
        "prior_orders": "REAL",
        "prior_chargebacks": "REAL",
        "discount_pct": "REAL",
        "shipping_speed": "TEXT",
        "hour_of_day": "REAL",
        "is_weekend": "INTEGER",
        "cart_size": "REAL",
        "merchant_risk_score": "REAL",
        "is_high_value": "INTEGER",
        "is_fraud": "INTEGER",
        "fraud_probability": "REAL",
    }
    for column, column_type in desired_columns.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE inference_logs ADD COLUMN {column} {column_type}")
    conn.commit()

ensure_inference_schema()

cursor.execute('''CREATE TABLE IF NOT EXISTS retraining_jobs
                (job_id TEXT PRIMARY KEY, 
                 start_time TEXT, 
                 status TEXT, 
                 champion_f1 REAL, 
                 challenger_f1 REAL, 
                 decision TEXT, 
                 shap_path TEXT,
                psi_score REAL,
                 ks_stat REAL,
                 ks_p REAL,
                 adwin_change INTEGER)''')
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
    # Fallback dummy model so the API can operate in demo environments without a real model
    class _DummyModel:
        def predict_proba(self, X):
            import numpy as _np
            # return a small probability influenced by price for demo purposes
            probs = []
            for _, row in X.iterrows():
                p = 0.01 + min(max((row.get('price', 0) - 10) / 2000.0, 0.0), 0.99)
                probs.append([1.0 - p, p])
            return _np.array(probs)

    model = _DummyModel()

# --- API Data Models ---
class InferenceRequest(BaseModel):
    user_id: float
    product_id: float
    price: float
    event_type: str
    category: str = "electronics"
    device_type: str = "desktop"
    channel: str = "organic"
    country: str = "US"
    session_duration: float = 0.0
    account_age_days: float = 0.0
    prior_orders: float = 0.0
    prior_chargebacks: float = 0.0
    discount_pct: float = 0.0
    shipping_speed: str = "standard"
    hour_of_day: float = 0.0
    is_weekend: int = 0
    cart_size: float = 0.0
    merchant_risk_score: float = 0.0
    is_high_value: bool = False
    is_fraud: bool = False

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
    features = pd.DataFrame([{
        'user_id': request.user_id,
        'product_id': request.product_id,
        'price': request.price,
        'event_type': request.event_type,
        'category': request.category,
        'device_type': request.device_type,
        'channel': request.channel,
        'country': request.country,
        'session_duration': request.session_duration,
        'account_age_days': request.account_age_days,
        'prior_orders': request.prior_orders,
        'prior_chargebacks': request.prior_chargebacks,
        'discount_pct': request.discount_pct,
        'shipping_speed': request.shipping_speed,
        'hour_of_day': request.hour_of_day,
        'is_weekend': request.is_weekend,
        'cart_size': request.cart_size,
        'merchant_risk_score': request.merchant_risk_score,
        'is_high_value': int(request.is_high_value)
    }])
    features = encode_feature_frame(features)
    
    # 2. Prediction
    # extract probability for the positive class (1: fraud)
    prob_fraud = float(model.predict_proba(features)[0][1])
    
    # 3. Logging to SQLite for Drift Detection
    timestamp = datetime.datetime.now().isoformat()
    try:
        # Adapt to whichever inference_logs schema exists in the DB
        cursor.execute("PRAGMA table_info(inference_logs)")
        cols = [r[1] for r in cursor.fetchall()]
        # Build insertion list matching existing columns
        values = []
        for c in cols:
            if c == 'timestamp':
                values.append(timestamp)
            elif c == 'user_id':
                values.append(request.user_id)
            elif c == 'product_id':
                values.append(request.product_id)
            elif c == 'price':
                values.append(request.price)
            elif c == 'event_type':
                values.append(request.event_type)
            elif c == 'category':
                values.append(request.category)
            elif c == 'device_type':
                values.append(request.device_type)
            elif c == 'channel':
                values.append(request.channel)
            elif c == 'country':
                values.append(request.country)
            elif c == 'session_duration':
                values.append(request.session_duration)
            elif c == 'account_age_days':
                values.append(request.account_age_days)
            elif c == 'prior_orders':
                values.append(request.prior_orders)
            elif c == 'prior_chargebacks':
                values.append(request.prior_chargebacks)
            elif c == 'discount_pct':
                values.append(request.discount_pct)
            elif c == 'shipping_speed':
                values.append(request.shipping_speed)
            elif c == 'hour_of_day':
                values.append(request.hour_of_day)
            elif c == 'is_weekend':
                values.append(request.is_weekend)
            elif c == 'cart_size':
                values.append(request.cart_size)
            elif c == 'merchant_risk_score':
                values.append(request.merchant_risk_score)
            elif c == 'is_high_value':
                values.append(request.is_high_value)
            elif c == 'is_fraud':
                values.append(request.is_fraud)
            elif c == 'fraud_probability':
                values.append(prob_fraud)
            else:
                values.append(None)

        placeholders = ','.join(['?'] * len(values))
        cursor.execute(f"INSERT INTO inference_logs ({','.join(cols)}) VALUES ({placeholders})", tuple(values))
        conn.commit()
    except Exception as e:
        print(f"Error logging to SQLite: {e}")
        
    return InferenceResponse(fraud_probability=prob_fraud)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
