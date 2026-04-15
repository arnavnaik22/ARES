#!/bin/bash

echo "🛑 Resetting ARES 2.0 Environment to Zero..."

# 1. Clean the SQLite Databases and WALs
echo "🧹 Wiping inference logs and retraining history..."
rm -f data/inference_logs.db
rm -f data/inference_logs.db-wal
rm -f data/inference_logs.db-shm

# 2. Clean SHAP Drift Images
echo "🧹 Deleting generated SHAP drift analysis plots..."
rm -f data/shap_drift_*.png

# 3. Clean Spark Checkpoints
echo "🧹 Wiping PySpark streaming checkpoints..."
rm -rf /tmp/ares_fresh_chkpt_v3

# 4. Wipe Kafka Queue (Docker Volumes)
echo "🧹 Destroying left-over Kafka messages..."
docker compose down -v
docker compose up -d

echo "✅ Environment Reset! You can now restart your terminals from scratch."
