"""
FastAPI Inference Service for ARES.
Loads the baseline XGBoost model and artifacts dynamically using a ModelManager.
Exposes POST /predict and GET /model-info endpoints, validates schemas, 
and logs inference requests to SQLite for drift monitoring.
"""

import os
import json
import sqlite3
import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

try:
    from src.feature_schema import encode_feature_frame, MODEL_FEATURES
except ImportError:
    from feature_schema import encode_feature_frame, MODEL_FEATURES

app = FastAPI(title="ARES Inference Service", version="1.0.0")

# --- Initialize SQLite Database ---
DB_PATH = "data/inference_logs.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute('PRAGMA journal_mode=WAL;')
cursor = conn.cursor()

# Ensure schema exists, including metadata columns
cursor.execute('''CREATE TABLE IF NOT EXISTS inference_logs
                (timestamp TEXT, user_id REAL, product_id REAL, price REAL, 
                 event_type TEXT, category TEXT, device_type TEXT, channel TEXT, country TEXT,
                 session_duration REAL, account_age_days REAL, prior_orders REAL, prior_chargebacks REAL,
                 discount_pct REAL, shipping_speed TEXT, hour_of_day REAL, is_weekend INTEGER, cart_size REAL,
                 merchant_risk_score REAL, is_high_value BOOLEAN, is_fraud BOOLEAN, fraud_probability REAL,
                 model_version TEXT)''')
conn.commit()

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
        "model_version": "TEXT",
    }
    for column, column_type in desired_columns.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE inference_logs ADD COLUMN {column} {column_type}")
    conn.commit()

ensure_inference_schema()

# Create retraining table if not exists
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


class ModelManager:
    """
    Manages loading the active model and related feature schema metadata from MLflow.
    Isolates model management from the FastAPI endpoint layers.
    """
    def __init__(self):
        self.model = None
        self.run_id = "dummy"
        self.dataset_name = "synthetic"
        self.adapter_class = "SyntheticAdapter"
        self.schema_version = "1.0.0"
        self.training_time = None
        
        self.model_features = []
        self.canonical_features = []
        self.predictive_features = []
        self.medians = {}
        self.feature_mapping = {}

    def load_active_model(self):
        """
        Query MLflow local tracking to find the latest finished model run.
        Download and parse schema and feature metadata.
        """
        print("ModelManager: Fetching latest active model from MLflow tracking...")
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
        client = MlflowClient()
        
        # Get all experiments starting with ARES_
        experiments = client.search_experiments()
        ares_exps = [e for e in experiments if e.name.startswith("ARES_")]
        
        if not ares_exps:
            print("ModelManager Warning: No experiments starting with 'ARES_' found. Using dummy model.")
            self._setup_dummy_model()
            return
            
        exp_ids = [e.experiment_id for e in ares_exps]
        runs = client.search_runs(experiment_ids=exp_ids, order_by=["start_time DESC"], max_results=20)
        
        latest_run = None
        for r in runs:
            if r.info.status == "FINISHED":
                # Check if model folder exists in artifacts
                latest_run = r
                break
                
        if latest_run is None:
            print("ModelManager Warning: No completed runs found. Using dummy model.")
            self._setup_dummy_model()
            return
            
        self.run_id = latest_run.info.run_id
        self.dataset_name = latest_run.data.params.get("dataset", "synthetic")
        self.adapter_class = latest_run.data.params.get("adapter_class", "SyntheticAdapter")
        self.schema_version = latest_run.data.params.get("feature_schema_version", "1.0.0")
        
        # Load timestamp
        try:
            start_time_ms = latest_run.info.start_time
            self.training_time = datetime.datetime.fromtimestamp(start_time_ms / 1000.0).isoformat()
        except Exception:
            self.training_time = None
            
        # Load XGBoost Model
        loaded_model = None
        for path in ["xgboost_baseline_model", "model"]:
            try:
                model_uri = f"runs:/{self.run_id}/{path}"
                loaded_model = mlflow.xgboost.load_model(model_uri)
                print(f"ModelManager: Successfully loaded XGBoost model from run: {self.run_id}")
                break
            except Exception:
                continue
                
        if loaded_model is None:
            print("ModelManager Warning: Model artifact load failed. Using dummy model.")
            self._setup_dummy_model()
            return
            
        self.model = loaded_model
        
        # Load feature artifacts from MLflow
        # 1. model_features.json (all columns in the exact training order)
        try:
            feat_path = client.download_artifacts(self.run_id, "model_features.json")
            with open(feat_path, 'r') as f:
                self.model_features = json.load(f)
            print(f"ModelManager: Loaded {len(self.model_features)} model features.")
        except Exception:
            # Fallback to canonical_features_used.json
            try:
                feat_path = client.download_artifacts(self.run_id, "canonical_features_used.json")
                with open(feat_path, 'r') as f:
                    self.model_features = json.load(f)
                print(f"ModelManager: Fallback loaded {len(self.model_features)} features from canonical_features_used.json.")
            except Exception:
                # Default to ARES canonical features
                self.model_features = MODEL_FEATURES.copy()
                print("ModelManager: Fallback loaded MODEL_FEATURES schema as model features.")
                
        # 2. canonical_features.json
        try:
            path = client.download_artifacts(self.run_id, "canonical_features.json")
            with open(path, 'r') as f:
                self.canonical_features = json.load(f)
        except Exception:
            # Infer: any feature in model_features that is in MODEL_FEATURES
            self.canonical_features = [f for f in self.model_features if f in MODEL_FEATURES]
            
        # 3. predictive_features.json
        try:
            path = client.download_artifacts(self.run_id, "predictive_features.json")
            with open(path, 'r') as f:
                self.predictive_features = json.load(f)
        except Exception:
            # Infer: any feature in model_features that is NOT in MODEL_FEATURES
            self.predictive_features = [f for f in self.model_features if f not in MODEL_FEATURES]
            
        # 4. adapter_metadata.json to get medians
        try:
            meta_path = client.download_artifacts(self.run_id, "adapter_metadata.json")
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.medians = metadata.get("medians", {})
            print(f"ModelManager: Loaded default medians for {len(self.medians)} columns.")
        except Exception:
            self.medians = {}

        # 5. Load feature_mapping.json
        try:
            map_path = client.download_artifacts(self.run_id, "feature_mapping.json")
            with open(map_path, 'r') as f:
                self.feature_mapping = json.load(f)
            print(f"ModelManager: Loaded feature mapping for {len(self.feature_mapping)} keys.")
        except Exception:
            self.feature_mapping = {}

    def _setup_dummy_model(self):
        """Sets up a dummy model for fallback in environments without active runs."""
        class _DummyModel:
            def predict_proba(self, X):
                probs = []
                for _, row in X.iterrows():
                    p = 0.01 + min(max((row.get('price', 0) - 10) / 2000.0, 0.0), 0.99)
                    probs.append([1.0 - p, p])
                return np.array(probs)
        self.model = _DummyModel()
        self.run_id = "dummy"
        self.model_features = MODEL_FEATURES.copy()
        self.canonical_features = MODEL_FEATURES.copy()
        self.predictive_features = []
        self.medians = {}
        print("ModelManager: Fallback setup dummy model complete.")


# Initialize Model Manager
model_manager = ModelManager()

@app.on_event("startup")
def startup_event():
    model_manager.load_active_model()


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


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if model_manager.model is None:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
         
    # 1. Validation of incoming canonical features and data types
    try:
        # Pydantic checks values automatically, but we also enforce type verification manually:
        for field_name, field_value in request.dict().items():
            if field_name in {'user_id', 'product_id', 'price', 'session_duration', 'account_age_days', 
                              'prior_orders', 'prior_chargebacks', 'discount_pct', 'hour_of_day', 
                              'is_weekend', 'cart_size', 'merchant_risk_score'}:
                if not isinstance(field_value, (int, float)):
                    raise ValueError(f"Field '{field_name}' must be numeric.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Request Validation Error: {e}")

    # 2. Construct combined feature vectors (canonical + predictive)
    req_dict = request.dict()
    
    # Check if all required active model features can be satisfied
    missing_required = []
    feature_vector = {}
    
    # Reverse map for convenience: raw column -> canonical key
    reverse_mapping = {}
    for canonical_k, raw_v in model_manager.feature_mapping.items():
        reverse_mapping[raw_v] = canonical_k
        
    for feat in model_manager.model_features:
        if feat in req_dict:
            feature_vector[feat] = req_dict[feat]
        elif feat in reverse_mapping and reverse_mapping[feat] in req_dict:
            feature_vector[feat] = req_dict[reverse_mapping[feat]]
        elif feat in model_manager.medians:
            feature_vector[feat] = model_manager.medians[feat]
        else:
            missing_required.append(feat)
            
    if missing_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Validation Error: The deployed model requires predictive features "
                   f"{missing_required} which are not supplied in the request and have no default values."
        )

    # Convert mapping vector to dataframe
    features_df = pd.DataFrame([feature_vector])
    
    # Process ARES standard encoding on canonical columns, keeping predictive raw
    features_canonical_encoded = encode_feature_frame(features_df)
    X_canonical = features_canonical_encoded[model_manager.canonical_features]
    X_predictive = features_df[model_manager.predictive_features]
    X = pd.concat([X_canonical, X_predictive], axis=1)
    
    # Slice using strict training ordering
    X = X[model_manager.model_features]
    
    # 3. Model Prediction
    start_time = datetime.datetime.now()
    prob_fraud = float(model_manager.model.predict_proba(X)[0][1])
    latency = (datetime.datetime.now() - start_time).total_seconds()
    
    # 4. SQLite Logging (Standard ARES Canonical Features only)
    timestamp = datetime.datetime.now().isoformat()
    try:
        cursor.execute("PRAGMA table_info(inference_logs)")
        cols = [r[1] for r in cursor.fetchall()]
        
        values = []
        for c in cols:
            if c == 'timestamp':
                values.append(timestamp)
            elif c == 'fraud_probability':
                values.append(prob_fraud)
            elif c == 'model_version':
                values.append(model_manager.run_id)
            elif c in req_dict:
                val = req_dict[c]
                # convert boolean to int
                if isinstance(val, bool):
                    val = int(val)
                values.append(val)
            else:
                values.append(None)

        placeholders = ','.join(['?'] * len(values))
        cursor.execute(f"INSERT INTO inference_logs ({','.join(cols)}) VALUES ({placeholders})", tuple(values))
        conn.commit()
    except Exception as e:
        print(f"Error logging to SQLite: {e}")
        
    return InferenceResponse(fraud_probability=prob_fraud)


@app.get("/model-info")
def model_info():
    """
    Exposes model metadata and feature counts for debugging and system monitoring.
    """
    if model_manager.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded.")
        
    return {
        "model_run_id": model_manager.run_id,
        "dataset_name": model_manager.dataset_name,
        "adapter_class": model_manager.adapter_class,
        "feature_schema_version": model_manager.schema_version,
        "training_timestamp": model_manager.training_time,
        "canonical_features_count": len(model_manager.canonical_features),
        "predictive_features_count": len(model_manager.predictive_features),
        "model_features_count": len(model_manager.model_features)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
