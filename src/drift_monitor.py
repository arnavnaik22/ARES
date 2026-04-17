"""
Drift Detection Engine for ARES.
Continuously calculates the Population Stability Index (PSI) of the 'price' feature
using the streaming inference logs compared to the static baseline data.
"""

import sqlite3
import pandas as pd
import numpy as np
import time

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
        expected_price = baseline_df['price'].dropna().values
        print(f"Baseline loaded. {len(expected_price)} records found.")
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
             query = "SELECT price FROM inference_logs ORDER BY timestamp DESC LIMIT 200"
             recent_batch = pd.read_sql_query(query, conn)
             conn.close()
             
             if len(recent_batch) > 10:
                  actual_price = recent_batch['price'].values
                  psi = calculate_psi(expected_price, actual_price)
                  
                  if psi > psi_threshold:
                       print(f"\nCONCEPT DRIFT DETECTED")
                       print(f"Feature: 'price' | Current PSI: {psi:.4f} (Threshold: {psi_threshold})")
                       print(f"Warning: The incoming distribution has significantly shifted from the baseline!\n")
                       
                       import uuid
                       import datetime
                       import subprocess
                       
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
                               cursor_chk.execute("INSERT INTO retraining_jobs (job_id, start_time, status, psi_score) VALUES (?, ?, ?, ?)", (job_id, start_time, 'PENDING', psi))
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
