"""
Kafka Producer for ARES.
Reads the synthetic ecommerce behavior dataset and publishes each row as a JSON payload to a Kafka topic.
"""

import os
import json
import time
import argparse
import pandas as pd
from kafka import KafkaProducer

try:
    from src.feature_schema import MODEL_FEATURES
except ImportError:
    from feature_schema import MODEL_FEATURES

def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """
    Initialize and return a KafkaProducer instance.
    """
    return KafkaProducer(
        bootstrap_servers=[bootstrap_servers],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def main():
    parser = argparse.ArgumentParser(description="ARES Stream Producer Simulator")
    parser.add_argument("--data_path", type=str, default="data/raw/ecommerce_behavior.csv", help="Path to the raw CSV file")
    parser.add_argument("--topic", type=str, default="ecommerce-events", help="Kafka topic to publish to")
    parser.add_argument("--broker", type=str, default="localhost:9092", help="Kafka bootstrap broker address")
    parser.add_argument("--delay", type=float, default=0.05, help="Sleep delay in seconds between events")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    print(f"Loading data from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Failed to load dataframe: {e}")
        return

    print(f"Connecting to Kafka broker at {args.broker}...")
    try:
        producer = create_producer(args.broker)
    except Exception as e:
        print(f"Failed to connect to Kafka producer: {e}. Check if Kafka is running.")
        return

    print(f"Starting stream to topic '{args.topic}'. Press Ctrl+C to stop.")
    try:
        for index, row in df.iterrows():
            payload = row.to_dict()

            cycle = index % 360
            stealth_window = 150 <= cycle < 220
            drift_window = cycle >= 220

            if stealth_window or drift_window:
                payload['price'] = round(float(payload.get('price', 0.0)) * (1.45 if stealth_window else 2.35) + (35.0 if stealth_window else 120.0), 2)
                payload['event_type'] = 'cart' if stealth_window else 'purchase'
                payload['device_type'] = 'mobile'
                payload['channel'] = 'affiliate' if stealth_window else 'social'
                payload['country'] = 'IN' if stealth_window else 'BR'
                payload['shipping_speed'] = 'overnight'
                payload['session_duration'] = max(float(payload.get('session_duration', 0.0)) * (0.72 if stealth_window else 0.55), 8.0)
                payload['discount_pct'] = max(float(payload.get('discount_pct', 0.0)) * (0.65 if stealth_window else 0.45), 0.0)
                payload['prior_chargebacks'] = int(payload.get('prior_chargebacks', 0)) + (1 if stealth_window else 2)
                payload['merchant_risk_score'] = min(float(payload.get('merchant_risk_score', 0.0)) + (0.14 if stealth_window else 0.28), 1.0)
                payload['account_age_days'] = max(int(payload.get('account_age_days', 0)) // (3 if stealth_window else 5), 1)
                payload['cart_size'] = int(payload.get('cart_size', 0)) + (1 if stealth_window else 3)
                payload['is_fraud'] = True
            else:
                payload['price'] = float(payload.get('price', 0.0))
                payload['session_duration'] = float(payload.get('session_duration', 0.0))
                payload['discount_pct'] = float(payload.get('discount_pct', 0.0))
                payload['merchant_risk_score'] = float(payload.get('merchant_risk_score', 0.0))
                payload['prior_chargebacks'] = int(payload.get('prior_chargebacks', 0))
                payload['cart_size'] = int(payload.get('cart_size', 0))
                payload['is_fraud'] = bool(payload.get('is_fraud', False))

            payload['is_high_value'] = bool(payload.get('is_high_value', float(payload['price']) > 500.0))
            payload['is_weekend'] = int(payload.get('is_weekend', 0))
            payload['hour_of_day'] = int(payload.get('hour_of_day', 0))

            # Keep the payload aligned with the model schema while preserving extra raw columns for observability.
            allowed_keys = set(MODEL_FEATURES) | {'event_time', 'is_fraud'}
            payload = {key: value for key, value in payload.items() if key in allowed_keys}

            producer.send(args.topic, value=payload)
            drift_status = "[DRIFT BURST]" if drift_window else ("[STEALTH DRIFT]" if stealth_window else "[NORMAL]")
            print(f"{drift_status} Sent event {index}: {payload}")
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        # Ensure all pending messages are sent before shutting down
        producer.flush()
        producer.close()
        print("Producer cleanly shut down.")

if __name__ == "__main__":
    main()
