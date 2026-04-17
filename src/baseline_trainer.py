"""
Baseline Model Trainer for ARES.
Trains an XGBoost classifier on ecommerce behavior data and logs metrics using MLflow locally.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the provided file path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical features.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing features like 'price', 'event_type'.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for modeling.
    """
    df_processed = df.copy()
    
    # 1. Handle missing values
    if 'is_fraud' in df_processed.columns:
        df_processed = df_processed.dropna(subset=['is_fraud'])
    
    # Impute continuous features (e.g. price) with median
    if 'price' in df_processed.columns:
        median_price = df_processed['price'].median()
        df_processed['price'] = df_processed['price'].fillna(median_price)
        
    # Impute categorical features (e.g. event_type) with mode
    if 'event_type' in df_processed.columns:
        mode_event = df_processed['event_type'].mode()[0]
        df_processed['event_type'] = df_processed['event_type'].fillna(mode_event)
        
    # Fill missing IDs with a placeholder value
    for col in ['product_id', 'user_id']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(-1)

    # 2. Encode categorical variables
    encoder = LabelEncoder()
    if 'event_type' in df_processed.columns:
        df_processed['event_type'] = encoder.fit_transform(df_processed['event_type'].astype(str))
        
    return df_processed

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> XGBClassifier:
    """
    Train an XGBoost classifier with given parameters.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        params (dict): Hyperparameters for the XGBClassifier.
        
    Returns:
        XGBClassifier: The trained model.
    """
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the XGBoost model on the test dataset.
    
    Args:
        model (XGBClassifier): Trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        
    Returns:
        dict: A dictionary containing 'accuracy', 'f1_score', and 'roc_auc' metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ARES Baseline XGBoost Trainer")
    parser.add_argument("--data_path", type=str, default="data/raw/ecommerce_behavior.csv", 
                        help="Path to the raw dataset CSV")
    parser.add_argument("--n_estimators", type=int, default=100, help="XGBoost n_estimators")
    parser.add_argument("--max_depth", type=int, default=4, help="XGBoost max_depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="XGBoost learning_rate")
    args = parser.parse_args()

    # Configure MLflow to track locally in the current directory (creates ./mlruns)
    mlflow.set_experiment("ARES_Phase1_Baseline")

    print(f"Loading data from {args.data_path}...")
    try:
        df = load_data(args.data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you place the dataset at the correct path before running the script.")
        return

    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Ensure target column is present
    if 'is_fraud' not in df_processed.columns:
        print("Error: Target column 'is_fraud' not found in dataset!")
        return

    features = ['user_id', 'event_type', 'product_id', 'price']
    
    # Keep only features actually present in the dataframe
    features = [f for f in features if f in df_processed.columns]
    
    X = df_processed[features]
    y = df_processed['is_fraud']
    
    # 80/20 train-test split stratifying on target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42
    }
    
    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # 1. Log parameters
        mlflow.log_params(params)
        
        # 2. Train model
        print(f"Training model with params: {params}...")
        model = train_model(X_train, y_train, params)
        
        # 3. Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        print("\n--- Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
            
        # 4. Log metrics
        mlflow.log_metrics(metrics)
        
        # 5. Log model artifact
        mlflow.xgboost.log_model(model, "xgboost_baseline_model")
        
        print("\nRun completed and logged to local MLflow.")

if __name__ == "__main__":
    main()
