"""
ARES Command Center Dashboard.
Real-time Streamlit dashboard to monitor streaming transactions, active model versions,
and the latest SHAP explainability drift analysis.
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

st.set_page_config(page_title="ARES Command Center", layout="wide")

# Auto-refresh the dashboard every 4 seconds safely to prevent aggressive UI dimming 
st_autorefresh(interval=4000, limit=None, key="ares_refresh")

def get_total_transactions():
    """Fetch total number of evaluated streaming events."""
    try:
        conn = sqlite3.connect("data/inference_logs.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inference_logs")
        total = cursor.fetchone()[0]
        conn.close()
        return total
    except Exception:
        return 0

def get_recent_transactions(limit=100):
    """Fetch the latest streaming events to display in the live feed."""
    try:
        conn = sqlite3.connect("data/inference_logs.db")
        df = pd.read_sql_query(f"SELECT * FROM inference_logs ORDER BY timestamp DESC LIMIT {limit}", conn)
        conn.close()
        # Convert timestamp to datetime for better plotting
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()

def get_active_model_version():
    """Query MLflow registry to find the current active model version."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("ARES_Phase1_Baseline")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], 
                order_by=["start_time DESC"], 
                max_results=10
            )
            for run in runs:
                if run.info.status == "FINISHED":
                    stage = run.data.tags.get("stage", "Baseline")
                    model_type = run.data.params.get("model_type", "Initial Baseline")
                    import datetime
                    dt = datetime.datetime.fromtimestamp(run.info.start_time / 1000.0)
                    version_str = dt.strftime("v%m.%d.%H%M")
                    clean_model_type = model_type.replace('_', ' ').title()
                    
                    return f"{clean_model_type} Release ({version_str})"
    except Exception:
        pass
    return "Not Available"

def get_retraining_jobs():
    """Fetch recent retraining jobs from the database."""
    try:
        conn = sqlite3.connect("data/inference_logs.db", timeout=10)
        conn.execute('PRAGMA journal_mode=WAL;')
        df = pd.read_sql_query("SELECT * FROM retraining_jobs ORDER BY start_time DESC", conn)
        conn.close()
        if not df.empty:
            df['start_time'] = pd.to_datetime(df['start_time'])
        return df
    except Exception:
        return pd.DataFrame()

def main():
    st.title("ARES Command Center")
    st.markdown("### Autonomous Machine Learning Reliability Platform — Phase 5")
    
    st.markdown("---")

    # 1. System Status Metrics
    st.header("Real-Time System Metrics")
    col1, col2, col3 = st.columns(3)
    
    total_tx = get_total_transactions()
    active_model = get_active_model_version()
    df_recent = get_recent_transactions(100)
    
    with col1:
        st.metric("Total Streamed Transactions", f"{total_tx:,}")
    with col2:
        st.metric("Current Production Model", active_model)
    with col3:
        high_risk_count = 0
        if not df_recent.empty:
            high_risk_count = len(df_recent[df_recent['fraud_probability'] > 0.5])
        st.metric("Recent High-Risk Transactions (Last 100)", high_risk_count, delta_color="inverse")
    
    st.markdown("---")

    # Layout: 2 Columns for Live Data feed & Visualizations
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("Live Activity Feed")
        st.write("Most recent transactions evaluated by the Inference API.")
        
        if not df_recent.empty:
            display_df = df_recent.head(15).copy()
            
            # Format datetime safely
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]
            
            # Highlight high fraud probabilities
            def highlight_fraud(row):
                is_fraud = row['fraud_probability'] > 0.5
                color = 'background-color: rgba(255, 75, 75, 0.2)' if is_fraud else ''
                return [color] * len(row)
            
            format_dict = {
                "price": "${:.2f}",
                "fraud_probability": "{:.2%}"
            }
            
            st.dataframe(
                display_df.style.apply(highlight_fraud, axis=1).format(format_dict), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No data received yet. Ensure `docker-compose` and `spark_processor.py` are running.")
            
    with right_col:
        st.subheader("Continuous Explainability & Drift")
        
        # Interactive Scatter Plot
        if not df_recent.empty:
            fig = px.scatter(
                df_recent.head(50), 
                x="timestamp", 
                y="fraud_probability", 
                color="event_type",
                title="Real-Time Fraud Probability (Last 50 Events)",
                labels={"timestamp": "Time", "fraud_probability": "Fraud Risk Probability", "event_type": "Event Type"},
                color_discrete_map={"cart": "#f6c342", "purchase": "#21a366", "view": "#4e73df"}
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Fraud Threshold", annotation_position="bottom right")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        with st.container(border=True):
            st.markdown("### Autonomous Self-Healing Pipeline")
            st.write("ARES continuously calculates **Population Stability Index (PSI)** to detect concept drift in real time. When anomalous data distribution is detected, the system automatically triggers an asynchronous Model Retraining Execution without interrupting inference.")
            
            jobs_df = get_retraining_jobs()
        
            if not jobs_df.empty:
                active_jobs = jobs_df[~jobs_df['status'].isin(['COMPLETED', 'FAILED', 'REJECTED'])]
                last_job = jobs_df.iloc[0]
                
                # --- Active Pipeline Visualization ---
                if not active_jobs.empty:
                    st.markdown("#### Active Retraining Execution")
                    active_job = active_jobs.iloc[0]
                    status_val = active_job['status']
                    
                    # Create a dynamic progress pipeline
                    stages = ['PENDING', 'STARTED', 'SHAP_ANALYSIS', 'TRAINING_CHALLENGER', 'COMPARING', 'COMPLETED']
                    try:
                        pct_complete = (stages.index(status_val) + 1) / len(stages)
                    except ValueError:
                        pct_complete = 0.5
                        
                    st.progress(pct_complete)
                    st.warning(f"**Step {int(pct_complete * 100)}%:** Currently executing `{status_val}`...")
                    
                    psi = active_job.get('psi_score')
                    if pd.notnull(psi):
                        st.error(f"**Trigger Cause:** Massive Drift Detected (PSI: {psi:.2f} > 0.20 Threshold)")
                
                # --- Final Comparison Results ---
                elif last_job['status'] == 'COMPLETED':
                    st.markdown("#### Latest Champion vs Challenger Evaluation")
                    
                    champ_f1 = last_job.get('champion_f1', 0)
                    chall_f1 = last_job.get('challenger_f1', 0)
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Detected PSI Drift", f"{last_job.get('psi_score', 0):.2f}", delta="Alert Threshold: 0.20", delta_color="inverse")
                    with m2:
                        st.metric("Champion F1 Score", f"{champ_f1:.4f}")
                    with m3:
                        st.metric("Challenger F1 Score", f"{chall_f1:.4f}", delta=f"{(chall_f1 - champ_f1):.4f} Margin")
                    
                    st.markdown("---")
                    if last_job['decision'] == 'CHALLENGER_WINS':
                        st.success("**Autonomous Rotation Executed:** The newly trained Challenger model significantly outperformed the production Champion. The API has seamlessly rotated without downtime.")
                    else:
                        st.info("**Champion Maintained:** The newly trained Challenger model was rejected. The current production Champion is more robust against the anomalous data.")
    
                st.markdown("---")
                
                # Display latest SHAP explicitly inside an organized container
                completed_jobs = jobs_df[jobs_df['shap_path'].notna()]
                if not completed_jobs.empty:
                    latest_shap = completed_jobs.iloc[0]['shap_path']
                    if os.path.exists(latest_shap):
                        with st.expander("View Root-Cause Diagnostic (XAI SHAP Analysis)", expanded=True):
                            st.markdown("When the system detected the drift spike, it autonomously interrogated its own model parameters using Shapley Additive Explanations. Here is exactly what features triggered the anomaly:")
                            st.image(latest_shap, caption=f"SHAP Summary Plot for Job {completed_jobs.iloc[0]['job_id']}")
                            
            else:
                st.info("**System Monitoring Active:** Stable data stream. Automated model training metrics will execute natively when the Population Stability Index exceeds 0.20.", icon=None)
                
                st.markdown("#### Latest Champion vs Challenger Evaluation")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Detected PSI Drift", "0.00", delta="Stable")
                with m2:
                    st.metric("Champion F1 Score", "N/A")
                with m3:
                    st.metric("Challenger F1 Score", "N/A")
                
                st.markdown("---")
                with st.expander("View Root-Cause Diagnostic (XAI SHAP Analysis)", expanded=False):
                    st.markdown("SHAP (Shapley Additive Explanations) visualizations will physically render here when the retraining engine autonomously interrogates gradient-boosted model parameters during a detected anomaly.")

if __name__ == "__main__":
    main()
