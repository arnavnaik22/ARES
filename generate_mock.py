import numpy as np
import pandas as pd
import os

# Ensures the raw data directory exists
os.makedirs('data/raw', exist_ok=True)

# Configurations
num_records = 60000
np.random.seed(42)

print(f"Generating {num_records} realistic eCommerce records...")

start_time = pd.Timestamp("2025-01-01")
interarrival_minutes = np.random.gamma(shape=1.2, scale=18.0, size=num_records)
event_time = start_time + pd.to_timedelta(np.cumsum(interarrival_minutes), unit="m")
hour_of_day = event_time.hour
is_weekend = (event_time.dayofweek >= 5).astype(int)

categories = np.array(["electronics", "fashion", "home", "beauty", "sports", "grocery", "travel", "luxury"])
category_weights = np.array([0.20, 0.18, 0.16, 0.12, 0.10, 0.10, 0.08, 0.06])
event_types = np.array(["view", "cart", "purchase"])
event_weights = np.array([0.62, 0.22, 0.16])
device_types = np.array(["desktop", "mobile", "tablet"])
device_weights = np.array([0.52, 0.38, 0.10])
channels = np.array(["organic", "email", "social", "paid_search", "affiliate"])
channel_weights = np.array([0.38, 0.18, 0.20, 0.16, 0.08])
countries = np.array(["US", "CA", "GB", "DE", "IN", "BR", "AE", "SG"])
country_weights = np.array([0.34, 0.13, 0.11, 0.10, 0.11, 0.10, 0.06, 0.05])
shipping_speeds = np.array(["standard", "expedited", "overnight", "pickup"])
shipping_weights = np.array([0.62, 0.21, 0.11, 0.06])

category = np.random.choice(categories, num_records, p=category_weights)
event_type = np.random.choice(event_types, num_records, p=event_weights)
device_type = np.random.choice(device_types, num_records, p=device_weights)
channel = np.random.choice(channels, num_records, p=channel_weights)
country = np.random.choice(countries, num_records, p=country_weights)
shipping_speed = np.random.choice(shipping_speeds, num_records, p=shipping_weights)

base_price = {
    "electronics": 320,
    "fashion": 95,
    "home": 140,
    "beauty": 58,
    "sports": 110,
    "grocery": 32,
    "travel": 410,
    "luxury": 780,
}
event_multiplier = np.where(event_type == "purchase", 1.22, np.where(event_type == "cart", 1.08, 0.96))
category_multiplier = np.vectorize(base_price.get)(category)
price = np.round(np.clip(category_multiplier * event_multiplier * np.random.lognormal(mean=0.0, sigma=0.35, size=num_records), 4.0, 2500.0), 2)

discount_pct = np.round(np.clip(np.random.beta(1.8, 7.5, size=num_records) + np.where(category == "luxury", -0.03, 0.0), 0.0, 0.75), 3)
account_age_days = np.clip(np.random.gamma(shape=2.4, scale=180.0, size=num_records), 5, 3650).astype(int)
prior_orders = np.clip(np.random.poisson(lam=np.interp(account_age_days, [5, 3650], [0.2, 24.0])), 0, 120)
prior_chargebacks = np.random.poisson(lam=np.clip(0.04 + (channel == "affiliate") * 0.12 + (country == "BR") * 0.08 + (account_age_days < 120) * 0.10, 0.01, 0.9))
session_duration = np.round(np.clip(np.random.lognormal(mean=np.log(420.0), sigma=0.55, size=num_records) * np.where(event_type == "purchase", 0.82, 1.0), 15.0, 6000.0), 1)
cart_size = np.clip(
    np.random.poisson(lam=1.0 + (event_type == "cart") * 2.2 + (event_type == "purchase") * 1.0 + (category == "grocery") * 0.6),
    0,
    18,
)
merchant_risk_score = np.clip(
    0.15
    + (category == "luxury") * 0.09
    + (category == "travel") * 0.08
    + (channel == "social") * 0.15
    + (channel == "affiliate") * 0.18
    + (device_type == "mobile") * 0.05
    + (country == "BR") * 0.10
    + (country == "IN") * 0.07
    + np.random.normal(0, 0.06, num_records),
    0.0,
    1.0,
)
is_high_value = (price > 500).astype(int)

fraud_signal = (
    -4.0
    + 0.0023 * price
    + 1.00 * (event_type == "purchase")
    + 0.55 * (event_type == "cart")
    + 0.55 * (device_type == "mobile")
    + 0.75 * (channel == "social")
    + 0.85 * (channel == "affiliate")
    + 0.70 * np.isin(country, ["BR", "IN", "AE"])
    + 0.60 * (shipping_speed == "overnight")
    + 0.90 * (discount_pct < 0.05)
    + 1.10 * (prior_chargebacks > 0)
    + 0.35 * (account_age_days < 60)
    + 0.30 * (session_duration < 120)
    + 0.25 * (cart_size >= 4)
    + 0.45 * merchant_risk_score
    - 0.30 * (prior_orders > 20)
    + 0.20 * is_weekend
    + np.random.normal(0, 0.85, num_records)
)
fraud_probability = 1.0 / (1.0 + np.exp(-fraud_signal))
is_fraud = np.random.binomial(1, np.clip(fraud_probability, 0.01, 0.99))

df = pd.DataFrame(
    {
        "event_time": event_time.astype(str),
        "user_id": np.random.randint(1000, 9000, num_records),
        "product_id": np.random.randint(100, 9999, num_records),
        "price": price,
        "event_type": event_type,
        "category": category,
        "device_type": device_type,
        "channel": channel,
        "country": country,
        "session_duration": session_duration,
        "account_age_days": account_age_days,
        "prior_orders": prior_orders,
        "prior_chargebacks": prior_chargebacks,
        "discount_pct": discount_pct,
        "shipping_speed": shipping_speed,
        "hour_of_day": hour_of_day.astype(int),
        "is_weekend": is_weekend.astype(int),
        "cart_size": cart_size.astype(int),
        "merchant_risk_score": np.round(merchant_risk_score, 3),
        "is_high_value": is_high_value,
        "is_fraud": is_fraud,
    }
)

# Save to the location
filepath = 'data/raw/ecommerce_behavior.csv'
df.to_csv(filepath, index=False)

print(f"Success! Data saved to {filepath}")
print(f"Total Fraud Cases: {df['is_fraud'].sum()} out of {num_records}")