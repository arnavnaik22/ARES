# ARES System Validation Benchmark Methodology

This document summarizes the validation methodology and concept drift scenarios used to benchmark the ARES platform's autonomous detection and recovery capabilities.

The live benchmark execution results and rolling timelines are saved in the generated outputs under [reports/synthetic_platform_validation/](file:///wsl.localhost/Ubuntu/home/eidolon/ARES2.0/reports/synthetic_platform_validation/).

---

## 1. Metric Calculations and Telemetry

During streaming transaction simulation, ARES evaluates system reliability and classifier quality continuously using rolling windows:

1.  **Population Stability Index (PSI)**: Measures shift in continuous feature distributions (primarily `price`) relative to the static training baseline:
    $$\text{PSI} = \sum \left( P_{\text{actual}} - P_{\text{expected}} \right) \times \ln\left(\frac{P_{\text{actual}}}{P_{\text{expected}}}\right)$$
    A PSI score above `0.20` indicates a significant distribution shift.
2.  **Rolling F1-Score**: Evaluates predictive quality over a sliding window of the last `50` transactions. The threshold is optimized dynamically on validation sets during retraining cycles.

---

## 2. Concept Drift Scenarios

ARES is evaluated across five distinct, reproducible, and probabilistic concept drift scenarios:

*   **Scenario A — Gradual Covariate Drift**: Gradually shifts desktop/US traffic to mobile devices (lowering risk to 0.05). Champion F1 drops gradually as mobile devices are falsely predicted as fraud due to historical baseline bias.
*   **Scenario B — Sudden Covariate Drift**: Suddenly shifts traffic to mobile devices (with risk 0.05) at Step 300. F1 degrades abruptly due to Champion false positives.
*   **Scenario C — Feature Distribution Drift**: Moderate 30% shift of device type to mobile at Step 300, leading to a smaller degradation and quick recovery.
*   **Scenario D — Concept Drift**: Shifts traffic to tablet devices at Step 300, and the classification rule flips so low-risk tablet purchases become fraud (concept shift). F1 drops as the Champion fails to flag new tablet fraud.
*   **Scenario E — Recurring Drift**: Combines Scenario B (covariate shift) followed by Scenario D (concept shift) to validate double consecutive closed-loop retraining and deployment cycles.

---

## 3. The Baseline-Degradation-Recovery Lifecycle

Every benchmark scenario tracks a clear lifecycle:
1.  **Baseline**: Model operates on reference data (F1-score stable, PSI < 0.1).
2.  **Drift Injected**: A drift scenario starts (Step 300).
3.  **PSI Increase**: Continuous statistical monitoring registers the shift.
4.  **Drift Detected**: PSI breaches the threshold (Step 500, recording latency).
5.  **Retraining Triggered**: Monitor registers a closed-loop Job ID in SQLite.
6.  **Challenger Deployed**: Retrained Challenger model is rolled out (Step 650).
7.  **Performance Recovered**: Evaluates the F1-score stabilization post-deployment.

---

## 4. Benchmark Generated Artifacts
*   **Markdown Summary Report**: [reports/synthetic_platform_validation/benchmark_report.md](file:///wsl.localhost/Ubuntu/home/eidolon/ARES2.0/reports/synthetic_platform_validation/benchmark_report.md)
*   **JSON Execution Telemetry**: [reports/synthetic_platform_validation/benchmark_report.json](file:///wsl.localhost/Ubuntu/home/eidolon/ARES2.0/reports/synthetic_platform_validation/benchmark_report.json)
*   **Performance Recovery Curves**: [reports/synthetic_platform_validation/recovery_curves/](file:///wsl.localhost/Ubuntu/home/eidolon/ARES2.0/reports/synthetic_platform_validation/recovery_curves/)
