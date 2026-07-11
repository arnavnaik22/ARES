# ARES Cleanup Report

This report documents the cleanup audit performed to prepare the ARES repository for production-quality public release.

---

## 1. Removed Root-Level Files

| File Name | Description / Rationale |
| :--- | :--- |
| `debug_client.py` | Temporary validation script used to verify TestClient API payload handling. |
| `debug_predictions.py` | Intermediate diagnostic script used to inspect classification probability thresholds. |
| `clean_up_summary.md` | Root-level markdown summary, superseded by this unified report in the documentation folder. |
| `mlflow.db.bak` | Legacy backup database file containing stale local runs. |
| `mlruns.bak/` | Legacy backup experiment directory containing deprecated MLflow runs. |
| `walkthrough.md` | Temporary pipeline walkthrough file containing obsolete checklists. |

---

## 2. Removed Obsolete Output Folders

| Directory Name | Description / Rationale |
| :--- | :--- |
| `reports/benchmark/` | Obsolete intermediate plot output directory from prior Scenario A-D benchmark validation runs. All scenario execution results are now unified under `reports/synthetic_platform_validation/`. |
| `reports/recovery/` | Legacy diagnostic curves from early single-scenario closed-loop pipeline runs, superseded by multi-scenario validations. |
| `reports/final_demo/` | Obsolete deployment pipeline timeline curves and system screenshots from early demonstration runs. |
| `evaluation/` | Obsolete pipeline metrics evaluation scripts left over from previous experiments. |
| `scratch/` | Temporary scratch scripts directory containing diagnostic metrics code. |

---

## 3. Preserved Assets and Dependencies

All baseline metrics, champion diagnostics, and deployment screenshots referenced by the Multi-Page Operations Console, `README.md`, or core documentation are preserved:
*   `reports/synthetic_platform_validation/`: Contains rolling recovery curves, PSI telemetry, and JSON summaries.
