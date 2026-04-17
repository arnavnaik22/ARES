import pandas as pd
import numpy as np
import os

# Ensures the raw data directory exists
os.makedirs('data/raw', exist_ok=True)

# Configurations
num_records = 15000
np.random.seed(42)

print(f"Generating {num_records} synthetic eCommerce records...")

data = {
    'user_id': np.random.randint(1000, 5000, num_records),
    'product_id': np.random.randint(100, 999, num_records),
    'price': np.round(np.random.uniform(5.0, 800.0, num_records), 2),
    'event_type': np.random.choice(['view', 'cart', 'purchase'], num_records, p=[0.75, 0.15, 0.10]),
}

df = pd.DataFrame(data)

# Higher-price purchases higher chance of being flagged as fraud
base_fraud_rate = np.random.choice([0, 1], num_records, p=[0.97, 0.03])
pattern_fraud = np.where((df['price'] > 500) & (df['event_type'] == 'purchase'), 1, 0)

df['is_fraud'] = np.maximum(base_fraud_rate, pattern_fraud)

# Save to the location
filepath = 'data/raw/ecommerce_behavior.csv'
df.to_csv(filepath, index=False)

print(f"Success! Data saved to {filepath}")
print(f"Total Fraud Cases: {df['is_fraud'].sum()} out of {num_records}")