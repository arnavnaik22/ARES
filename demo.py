"""
ARES End-to-End Live Simulation and Recovery Console Launcher.
Starts the synthetic benchmark simulation in the background and automatically
launches the Streamlit Operations Console in the foreground.
"""

import os
import sys
import subprocess
import time
import json
import sqlite3
import threading
import datetime

def check_dependencies():
    print("Verifying ARES system packages...")
    packages = ["xgboost", "mlflow", "pandas", "fastapi", "uvicorn", "streamlit", "shap", "matplotlib"]
    missing = []
    for p in packages:
        try:
            __import__(p)
        except ImportError:
            missing.append(p)
    if missing:
        print(f"Error: Missing required packages: {missing}")
        print("Please install them using: pip install -r requirements.txt")
        sys.exit(1)
    print("Packages verified successfully.")

def check_mock_data():
    print("Verifying e-commerce transaction data...")
    if not os.path.exists("data/train_transaction.csv"):
        print("Mock dataset not found. Generating mock transaction records...")
        subprocess.run([sys.executable, "generate_mock.py"], check=True)
    print("Dataset verified.")

def monitor_benchmark(proc, log_filepath):
    # Monitor background process and report immediately on crash/failure
    while proc.poll() is None:
        time.sleep(0.5)
        
    return_code = proc.poll()
    if return_code is not None and return_code != 0:
        print(f"\n[ERROR] Background benchmark process died with exit code {return_code}.")
        print("Surfacing traceback/errors from log file (last 50 lines):")
        print("----------------------------------------------------------------------")
        if os.path.exists(log_filepath):
            try:
                with open(log_filepath, "r") as f:
                    content = f.read()
                    lines = content.splitlines()
                    for line in lines[-50:]:
                        print(line)
            except Exception as e:
                print(f"Error reading log file: {e}")
        print("----------------------------------------------------------------------")
        
        # Write failure state to live_status.json atomically so Streamlit dashboard reflects this
        try:
            status_file = "reports/synthetic_platform_validation/live_status.json"
            status_data = {
                "scenario": "Orchestration Error",
                "step": 0,
                "total_steps": 1100,
                "psi": 0.0,
                "f1": 0.0,
                "active_model": "None",
                "retraining_cycles": 0,
                "stage": "Failed",
                "detection_status": "Failed",
                "deployment_status": "Failed",
                "completed": True,
                "elapsed_time": 0.0,
                "last_updated": "N/A",
                "event_log": ["Simulation Failed", "Process died on background worker.", "See terminal logs for stack trace."],
                "timestamp": time.time()
            }
            tmp_file = status_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(status_data, f, indent=4)
            os.replace(tmp_file, status_file)
            
            # Append to trace log
            t_str = datetime.datetime.now().strftime("%H:%M:%S")
            with open("reports/synthetic_platform_validation/benchmark_state_trace.log", "a") as trace_f:
                trace_f.write(f"{t_str} | demo.py | Failed | step=0 | FAILED | process exited with code {return_code}\n")
        except Exception:
            pass

def main():
    check_dependencies()
    check_mock_data()

    # Clear old live status, SQLite databases, and trace logs to start fresh
    status_file = "reports/synthetic_platform_validation/live_status.json"
    if os.path.exists(status_file):
        try:
            os.remove(status_file)
        except Exception:
            pass
            
    trace_log = "reports/synthetic_platform_validation/benchmark_state_trace.log"
    if os.path.exists(trace_log):
        try:
            os.remove(trace_log)
        except Exception:
            pass
            
    try:
        conn = sqlite3.connect("data/inference_logs.db")
        conn.execute("DROP TABLE IF EXISTS inference_logs")
        conn.execute("DROP TABLE IF EXISTS retraining_jobs")
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Ensure reports folder exists
    os.makedirs("reports/synthetic_platform_validation", exist_ok=True)

    print("\nStarting multi-scenario concept drift benchmark simulation in the background...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Route background benchmark output to a log file
    log_filepath = "reports/synthetic_platform_validation/benchmark.log"
    log_file = open(log_filepath, "w")
    benchmark_proc = subprocess.Popen(
        [sys.executable, "src/run_synthetic_benchmark.py"],
        env=env,
        stdout=log_file,
        stderr=log_file
    )

    # Start the monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_benchmark,
        args=(benchmark_proc, log_filepath),
        daemon=True
    )
    monitor_thread.start()

    print("Launching Streamlit Operations Console dashboard in the foreground...")
    print("Press Ctrl+C to terminate both the dashboard and the background simulation.")
    
    try:
        # Run Streamlit dashboard in foreground
        streamlit_bin = os.path.join(os.path.dirname(sys.executable), "streamlit")
        if not os.path.exists(streamlit_bin):
            streamlit_bin = "streamlit"
        subprocess.run([streamlit_bin, "run", "dashboard/app.py"], env=env)
    except KeyboardInterrupt:
        print("\nShutting down ARES operations simulation...")
    finally:
        benchmark_proc.terminate()
        log_file.close()
        print("ARES simulation stopped.")

if __name__ == "__main__":
    main()
