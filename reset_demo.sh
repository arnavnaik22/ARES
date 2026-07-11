#!/bin/bash

echo "Resetting ARES 2.0 Environment to Zero..."

# 1. Recreate essential directories
mkdir -p data
mkdir -p reports/synthetic_platform_validation

# 2. Clean the SQLite Databases and WALs
echo "Wiping inference logs and retraining history..."
rm -f data/inference_logs.db
rm -f data/inference_logs.db-wal
rm -f data/inference_logs.db-shm

# 3. Clean SHAP Drift Images
echo "Deleting generated SHAP drift analysis plots..."
rm -f data/shap_drift_*.png

# 4. Reset MLflow tracking state
echo "Resetting MLflow tracking artifacts..."
rm -rf mlruns
rm -f mlflow.db
rm -f mlflow.db-wal
rm -f mlflow.db-shm

# 5. Wipe PySpark streaming checkpoints
echo "Wiping PySpark streaming checkpoints..."
rm -rf /tmp/ares_fresh_chkpt_v3

# 6. Wipe live status and trace logs
echo "Clearing live dashboard status and trace logs..."
rm -f reports/synthetic_platform_validation/live_status.json
rm -f reports/synthetic_platform_validation/live_status.json.tmp
rm -f reports/synthetic_platform_validation/benchmark_state_trace.log

# 7. Wipe Kafka Queue (Docker Volumes)
echo "Destroying left-over Kafka messages..."
if command -v docker &> /dev/null; then
    docker compose down -v
    docker compose up -d
else
    echo "Docker not found, skipping container reset."
fi

echo "Environment Reset!"
