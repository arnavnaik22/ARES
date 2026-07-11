# ARES Release Checklist

This document contains the release verification checklist for ARES v1.0.

---

## Verification Checklist

*   [x] **Demo works**: `python demo.py` executes cleanly, training the champion baseline, running the multi-scenario benchmark, outputting metrics, and launching the Streamlit dashboard automatically.
*   [x] **Dashboard works**: Multi-page operations console launches via `streamlit run dashboard/app.py` with unbroken rendering of timeline curves, PSI telemetry, and SHAP explainers.
*   [x] **MLflow works**: Champion and Challenger runs log models, features, default medians, and adapter mappings correctly. Local UI starts with `mlflow ui`.
*   [x] **FastAPI works**: Inference service boots up with `uvicorn src.inference_service:app` and exposes the `/predict` endpoint to process live transactional payloads.
*   [x] **Benchmark reproducible**: Multi-scenario validation script `src/run_synthetic_benchmark.py` completes warning-free, producing identical performance lifecycle tables.
*   [x] **Documentation complete**: Core specs, sequence flows, limits, and cleanup logs are consolidated inside the `docs/` folder.
*   [x] **README verified**: Landing page contains motivation, structure diagrams, stack tables, benchmark charts, and is readable under 3 minutes.
*   [x] **GitHub ready**: Contains standard `LICENSE`, `.gitignore`, `requirements.txt`, clean relative paths, and zero cache logs.
