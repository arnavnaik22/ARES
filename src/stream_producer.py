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
    parser.add_argument("--delay", type=float, default=0.5, help="Sleep delay in seconds between events")
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
            # Convert row to dictionary payload
            payload = row.to_dict()
            
            # Simulate dynamic Concept Drift
            # For every block of 150 events: 
            # - First 100: Normal baseline data
            # - Next 50: Severe concept drift (Prices surge, anomalous purchase behavior)
            if 100 <= (index % 150) <= 150:
                payload['price'] = (payload['price'] * 5.0) + 1000.0
                payload['event_type'] = 'purchase'
            else:
                # Keep normal values, but ensure float formatting
                payload['price'] = float(payload['price'])
                
            # Send the data asynchronously to the given topic
            producer.send(args.topic, value=payload)
            
            drift_status = "[DRIFT BURST]" if 100 <= (index % 150) <= 150 else "[NORMAL]"
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
