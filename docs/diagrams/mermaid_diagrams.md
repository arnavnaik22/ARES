# ARES Mermaid System Diagrams

This document contains polished Mermaid system architecture and flow diagrams representing the ARES ML Reliability Platform.

---

## 1. Overall System Architecture
```mermaid
graph TD
    A[Data Ingestion] --> B[Dataset Adapter Layer]
    B --> C[Baseline Trainer]
    C --> D[MLflow Model Registry]
    D --> E[FastAPI Inference Service]
    E --> F[SQLite Transaction Log]
    F --> G[Drift Monitor]
    G -- PSI Trigger --> H[Retraining Engine]
    H --> C
```

---

## 2. Training Pipeline
```mermaid
flowchart TD
    A[Raw Ingestion] --> B[Adapter Feature Separation]
    B --> C[Partition Data: Train/Val/Test]
    C --> D[Fit XGBoost Classifier]
    D --> E[Calibrate Threshold on Val]
    E --> F[Evaluate Offline on Test]
    F --> G[Upload Registry Runs to MLflow]
```

---

## 3. Inference Pipeline
```mermaid
flowchart TD
    A[API Post Request] --> B[Pydantic Validation]
    B --> C[ModelManager Median Imputation]
    C --> D[XGBoost Predict Probability]
    D --> E[Threshold Check: Fraud Risk Alert]
    E --> F[Log Transaction Payload to SQLite]
    F --> G[Return API JSON Response]
```

---

## 4. Drift Detection Pipeline
```mermaid
flowchart TD
    A[Streaming Logs Ingested] --> B[Sliding Window Ingestion]
    B --> C[Quantile Binning against Reference]
    C --> D[Compute Population Stability Index]
    D --> E{PSI > 0.20?}
    E -- Yes --> F[Register Retraining Job in SQLite]
    E -- No --> G[Continue Stream Monitoring]
```

---

## 5. Retraining Pipeline
```mermaid
flowchart TD
    A[Drift Trigger Received] --> B[Query SQLite Log Buffers]
    B --> C[Merge Baseline & Drift Sets]
    C --> D[Shuffle & Partition Splits]
    D --> E[Fit Challenger XGBoost]
    E --> F[Calibrate Challenger Threshold]
    F --> G[Evaluate offline Champion vs Challenger]
    G --> H[Upload Challenger to MLflow]
```

---

## 6. Model Lifecycle
```mermaid
stateDiagram-v2
    [*] --> Historical_Train
    Historical_Train --> Active_Champion : MLflow Deploy
    Active_Champion --> Streaming_Inference
    Streaming_Inference --> Drift_Detected : PSI > 0.20
    Drift_Detected --> Retrain_Challenger
    Retrain_Challenger --> Active_Challenger : Challenger Wins
    Active_Challenger --> Streaming_Inference
    Retrain_Challenger --> Active_Champion : Challenger Fails
```

---

## 7. Dataset Adapter Architecture
```mermaid
classDiagram
    class DatasetAdapter {
        +process(df_raw) AdapterOutput
    }
    class IEEECISAdapter {
        +process(df_raw) AdapterOutput
    }
    class SyntheticAdapter {
        +process(df_raw) AdapterOutput
    }
    class AdapterOutput {
        +df_processed: DataFrame
        +canonical_features: List
        +predictive_features: List
        +feature_mapping: Dict
        +medians: Dict
    }
    DatasetAdapter <|-- IEEECISAdapter
    DatasetAdapter <|-- SyntheticAdapter
    AdapterOutput <.. DatasetAdapter : returns
```

---

## 8. MLflow Integration
```mermaid
flowchart LR
    A[ARES Pipeline] --> B(MLflow Tracking Client)
    B --> C[Parameters: scale_pos_weight, learning_rate]
    B --> D[Metrics: F1, Precision, Recall, AUC]
    B --> E[Artifacts: feature_mapping.json, model_features.json]
    B --> F[Model weights: model.xgb]
```

---

## 9. Component Interaction
```mermaid
flowchart TD
    A[FastAPI Gateway] <--> B[ModelManager]
    B <--> C[MLflow Registry]
    A --> D[(SQLite logs)]
    E[Drift Monitor] --> D
    E --> F[Retraining Script]
    F --> C
```

---

## 10. Sequence Diagram
```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Inference
    participant DB as SQLite DB
    participant Mon as Drift Monitor
    participant Ret as Retraining Engine

    Client->>API: POST /predict (canonical features)
    API->>API: Impute raw features with medians
    API->>DB: Log transaction prediction
    API->>Client: Return prediction probability
    Mon->>DB: Query recent window
    Mon->>Mon: Calculate PSI
    Note over Mon: PSI crosses 0.20 threshold
    Mon->>DB: Write retraining job trigger
    Mon->>Ret: Invoke Retrainer
    Ret->>DB: Query drift logs
    Ret->>Ret: Train Challenger model
    Ret->>API: Rotates Model Manager active ID
```

---

## 11. Deployment Diagram
```mermaid
flowchart TD
    subgraph Client App
        A[Transaction Initiator]
    subgraph FastAPI Container
        B[REST Scoring Gateway]
    subgraph SQLite Instance
        C[(inference_logs.db)]
    subgraph MLflow Server
        D[Run Directory store]
    
    A -->|HTTP POST| B
    B -->|Logs evaluations| C
    B -->|Pulls active run| D
```
