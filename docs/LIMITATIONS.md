# ARES ML Reliability Platform: Limitations and Assumptions

This document outlines the current assumptions, known technical and scientific limitations, and proposed future improvements for the ARES platform. It is designed to provide transparent, technically defensible context for public presentation and system interviews.

---

## 1. Current Assumptions

*   **Probabilistic Label Generation in Simulations**: To test closed-loop drift detection and retraining realistically, our streaming simulations inject fraud labels using a probabilistic logistic model with random normal noise. This avoids near-perfect metrics (e.g. 1.00 F1) and yields realistic baseline, degradation, and recovery metrics.
*   **Stationary Reference Distribution**: The Population Stability Index (PSI) calculation assumes that the baseline dataset (reference distribution) is static. In practice, baseline distributions drift organically over time due to business changes or seasonal shifts.
*   **FastAPI Local Storage**: The API assumes a local SQLite instance is sufficient for storing transaction log buffers for drift detection and retraining.

---

## 2. Technical and Scientific Limitations

### A. API Featurization Mismatch Bottleneck
*   **Problem**: Models are trained on a Dual-Feature architecture combining ARES canonical features (7 variables) and raw dataset-specific predictive features (e.g., 310 raw columns from IEEE-CIS). However, the standardized FastAPI `/predict` endpoint only accepts the 20 ARES canonical features.
*   **Impact**: During streaming inference, the model manager is forced to impute all 310 raw predictive features with static medians loaded from the metadata. This flattens model discrimination, compresses predicted probabilities, and forces the model to act as a canonical-only classifier on the stream.
*   **Consequence**: The F1 score on the live stream is highly sensitive to threshold choices because probabilities cluster tightly around constant boundaries.

### B. Uncalibrated Probabilities via `scale_pos_weight`
*   **Problem**: To handle severe class imbalance, the XGBoost model is trained using `scale_pos_weight` (ratio of negative to positive classes).
*   **Impact**: While this helps the model prioritize positive class instances during gradient updates, it shifts the predicted probability outputs away from actual class frequencies, resulting in uncalibrated probabilities.
*   **Consequence**: A fixed decision threshold of `0.5` cannot be used. The threshold must be calibrated on validation splits, and probabilities cannot be interpreted directly as actual risk percentages without calibration (e.g., Platt scaling or Isotonic Regression).

### C. Rolling-Window Metric Volatility
*   **Problem**: In streaming drift validation, calculating ROC-AUC on small rolling windows (e.g. size 50) under high class imbalance (12.5% or 3.5% fraud rate) is mathematically volatile.
*   **Impact**: A window with only 0 or 1 positive labels yields invalid or wild ROC-AUC scores (e.g. `0.09` or `1.00`), creating false alarms in monitoring dashboards.

---

## 3. Future Improvements

1.  **Dynamic Schema Extension (Schema Registry)**:
    *   Implement a dynamic schema registry in the FastAPI Inference Service that allows adapters to register their custom raw predictive features. The API gateway can then accept and validate arbitrary dataset-specific payloads, avoiding median imputation.
2.  **Probability Calibration Layer**:
    *   Integrate a post-training calibration step using Platt scaling (Logistic Regression) or Isotonic Regression on held-out validation sets before registering models in MLflow. This ensures that output probabilities represent true empirical risk.
3.  **Dynamic Reference Baseline Updates**:
    *   Extend the drift monitor to update its reference baseline distributions periodically (e.g. daily/weekly rolling windows) to prevent organic distribution changes from triggering false drift alarms.
4.  **Robust Online Metric Tracking**:
    *   Instead of rolling-window ROC-AUC, track online metrics using cumulative metrics with decaying windows, or rely on segment-level evaluations triggered at specific ingestion checkpoints.
