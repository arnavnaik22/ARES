"""
Drift Detection Engine for ARES.
Continuously calculates the Population Stability Index (PSI) of the 'price' feature
using the streaming inference logs compared to the static baseline data.
"""

import sqlite3
import pandas as pd
import numpy as np
import time

MONITORED_FEATURES = ["price", "discount_pct", "session_duration", "merchant_risk_score"]

# Try to use river's ADWIN if available for adaptive window drift detection
try:
    from river.drift import ADWIN
except Exception:
    ADWIN = None

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate the Population Stability Index (PSI) using quantiles.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from the expected distribution
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    # Ensure wide enough bins at the edges to catch everything
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Calculate frequencies in each bucket
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (a_perc - e_perc) * np.log(a_perc / e_perc)

    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])
    return psi_value

def main():
    print("Initializing ARES Drift Monitor...")
    
    # 1. Load Baseline Data
    baseline_path = "data/raw/ecommerce_behavior.csv"
    try:
        baseline_df = pd.read_csv(baseline_path)
        print(f"Baseline loaded. {len(baseline_df)} records found.")
    except Exception as e:
         print(f"Failed to load baseline data: {e}. Check if data/raw/ecommerce_behavior.csv exists.")
         return
         
    db_path = "data/inference_logs.db"
    poll_interval = 10
    psi_threshold = 0.2
    
    print(f"Starting continuous monitoring loop (Polling every {poll_interval}s, PSI Alert > {psi_threshold})...")
    
    while True:
         try:
             conn = sqlite3.connect(db_path)
             query = f"SELECT {', '.join(MONITORED_FEATURES)} FROM inference_logs ORDER BY timestamp DESC LIMIT 200"
             recent_batch = pd.read_sql_query(query, conn)
             conn.close()
             
             if len(recent_batch) > 10:
                  psi_scores = []
                  for feature in MONITORED_FEATURES:
                      if feature not in recent_batch.columns or feature not in baseline_df.columns:
                          continue
                      expected_values = baseline_df[feature].dropna().values
                      actual_values = recent_batch[feature].dropna().values
                      if len(expected_values) > 0 and len(actual_values) > 0:
                          psi_scores.append(calculate_psi(expected_values, actual_values))

                  psi = float(np.mean(psi_scores)) if psi_scores else 0.0
                  
                  if psi > psi_threshold:
                       print(f"\nCONCEPT DRIFT DETECTED")
                       print(f"Features: {', '.join(MONITORED_FEATURES)} | Current PSI: {psi:.4f} (Threshold: {psi_threshold})")
                       print(f"Warning: The incoming distribution has significantly shifted from the baseline!\n")
                       
                       import uuid
                       import datetime
                       import subprocess
                       # ADWIN detection over the recent batch
                       adwin_change = False
                       try:
                           if ADWIN is not None and len(recent_batch) > 0:
                               ad = ADWIN()
                               for v in recent_batch['price'].values[::-1]:
                                   ad.update(float(v))
                                   if ad.change_detected:
                                       adwin_change = True
                                       break
                           else:
                               # Fallback heuristic: large mean shift
                               if 'price' in baseline_df.columns and len(baseline_df['price'].dropna()) > 0:
                                   mean_expected = np.mean(baseline_df['price'].dropna().values)
                                   mean_actual = np.mean(recent_batch['price'].dropna().values)
                                   if abs(mean_actual - mean_expected) / (mean_expected + 1e-9) > 0.25:
                                       adwin_change = True
                       except Exception as e:
                           print(f"ADWIN check failed: {e}")
                       
                       try:
                           conn_chk = sqlite3.connect(db_path, timeout=10)
                           conn_chk.execute('PRAGMA journal_mode=WAL;')
                           cursor_chk = conn_chk.cursor()
                           timeout_threshold = (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat()
                           cursor_chk.execute(f"UPDATE retraining_jobs SET status = 'FAILED', decision = 'ORPHANED_TIMEOUT' WHERE status NOT IN ('COMPLETED', 'FAILED', 'REJECTED') AND start_time < '{timeout_threshold}'")
                           conn_chk.commit()

                           # Check for legitimately unfinished jobs
                           cursor_chk.execute("SELECT job_id FROM retraining_jobs WHERE status NOT IN ('COMPLETED', 'FAILED', 'REJECTED')")
                           active_jobs = cursor_chk.fetchall()
                           
                           # Check for recent cooldown
                           cursor_chk.execute("SELECT start_time FROM retraining_jobs ORDER BY start_time DESC LIMIT 1")
                           last_job_time = cursor_chk.fetchone()
                           
                           on_cooldown = False
                           if last_job_time:
                               try:
                                   last_dt = datetime.datetime.fromisoformat(last_job_time[0])
                                   if (datetime.datetime.now() - last_dt).total_seconds() < 40:
                                       on_cooldown = True
                               except Exception:
                                   pass
                           
                           if len(active_jobs) > 0:
                               print(f"[INFO] A retraining job ({active_jobs[0][0]}) is already active. Skipping new trigger.")
                           elif on_cooldown:
                               print(f"[COOLDOWN] System is cooling down to let the stream baseline clear. Skipping trigger.")
                           else:
                               job_id = f"job_{uuid.uuid4().hex[:8]}"
                               start_time = datetime.datetime.now().isoformat()
                               cursor_chk.execute("INSERT INTO retraining_jobs (job_id, start_time, status, psi_score, adwin_change) VALUES (?, ?, ?, ?, ?)", (job_id, start_time, 'PENDING', psi, int(adwin_change)))
                               conn_chk.commit()
                               
                               import sys
                               print(f"[INFO] Spawning background Retraining Engine (Job ID: {job_id})")
                               # Spawn detached process using same python executable
                               subprocess.Popen([sys.executable, "-m", "src.retraining_engine", "--job_id", job_id])
                       except Exception as e:
                           print(f"[ERROR] Failed to spawn retraining job: {e}")
                       finally:
                           conn_chk.close()
                  else:
                       print(f"[OK] Drift Monitor checking in. Current PSI: {psi:.4f}")
             else:
                  print("Waiting for more inference logs to calculate PSI...")
                  
         except sqlite3.OperationalError:
             print("Waiting for SQLite database to be created by the Inference Service...")
         except Exception as e:
             print(f"Error during drift calculation: {e}")
             
         time.sleep(poll_interval)

if __name__ == "__main__":
    main()
