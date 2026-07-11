# ARES Directory Layout

This document describes the structure of the ARES repository for public release.

```text
ARES2.0/
├── demo.py                        # One-command demo entry point
├── reset_demo.sh                  # Utility script to clean up DBs and MLflow logs
├── generate_mock.py               # Script to generate raw transactional dataset
├── docker-compose.yml             # Docker config to launch Zookeeper & Kafka Broker
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
├── README.md                      # Project landing page
│
├── docs/                          # Core system documentation
│   ├── ARCHITECTURE.md            # High-level architecture specs and UML layout
│   ├── TECHNICAL_DOCUMENTATION.md # Unified platform specifications
│   ├── LIMITATIONS.md             # Model assumptions and constraints
│   ├── BENCHMARKS.md              # Concept drift scenario methodology
│   ├── CLEANUP_REPORT.md          # Audit report of deleted files
│   ├── RELEASE_CHECKLIST.md       # Release verification checklist
│   └── REPOSITORY_STRUCTURE.md    # Repository tree overview (this file)
│
├── dashboard/                     # Multi-page interactive Streamlit app
│   └── app.py
│
├── reports/                       # Preserved evaluation artifacts and outputs
│   └── synthetic_platform_validation/ # Unified validation logs and curves
│
└── src/                           # Platform source modules
    ├── feature_schema.py          # Feature schema definition and encoding
    ├── baseline_trainer.py        # Champion baseline model trainer
    ├── inference_service.py       # FastAPI REST API serving prediction inputs
    ├── drift_monitor.py           # Stream monitoring and PSI alerts
    ├── retraining_engine.py       # Retraining process worker
    ├── spark_processor.py         # PySpark Structured Streaming processor
    ├── stream_producer.py         # Kafka transaction event stream producer
    └── run_synthetic_benchmark.py # Multi-scenario drift validation benchmark suite
```
