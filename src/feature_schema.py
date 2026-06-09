"""Shared feature schema and encoding helpers for ARES demo data."""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "user_id",
    "product_id",
    "price",
    "session_duration",
    "account_age_days",
    "prior_orders",
    "prior_chargebacks",
    "discount_pct",
    "hour_of_day",
    "cart_size",
    "merchant_risk_score",
    "is_high_value",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "event_type",
    "category",
    "device_type",
    "channel",
    "country",
    "shipping_speed",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_MAPS = {
    "event_type": {"view": 0, "cart": 1, "purchase": 2},
    "category": {
        "electronics": 0,
        "fashion": 1,
        "home": 2,
        "beauty": 3,
        "sports": 4,
        "grocery": 5,
        "travel": 6,
        "luxury": 7,
    },
    "device_type": {"desktop": 0, "mobile": 1, "tablet": 2},
    "channel": {"organic": 0, "email": 1, "social": 2, "paid_search": 3, "affiliate": 4},
    "country": {"us": 0, "ca": 1, "gb": 2, "de": 3, "in": 4, "br": 5, "ae": 6, "sg": 7},
    "shipping_speed": {"standard": 0, "expedited": 1, "overnight": 2, "pickup": 3},
}

DEFAULT_CATEGORY_VALUES = {column: max(mapping.values()) for column, mapping in CATEGORY_MAPS.items()}
DEFAULT_NUMERIC_VALUES = {
    "user_id": 0,
    "product_id": 0,
    "price": 0.0,
    "session_duration": 0.0,
    "account_age_days": 0,
    "prior_orders": 0,
    "prior_chargebacks": 0,
    "discount_pct": 0.0,
    "hour_of_day": 0,
    "cart_size": 0,
    "merchant_risk_score": 0.0,
    "is_high_value": 0,
    "is_weekend": 0,
}


def encode_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready dataframe with stable numeric encodings."""

    encoded = frame.copy()

    for column in NUMERIC_FEATURES:
        if column not in encoded.columns:
            encoded[column] = DEFAULT_NUMERIC_VALUES[column]
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(DEFAULT_NUMERIC_VALUES[column])

    if "is_high_value" not in frame.columns:
        encoded["is_high_value"] = (encoded["price"] > 500).astype(int)
    else:
        encoded["is_high_value"] = pd.to_numeric(encoded["is_high_value"], errors="coerce").fillna(0).astype(int)

    if "is_weekend" not in frame.columns:
        encoded["is_weekend"] = 0
    else:
        encoded["is_weekend"] = pd.to_numeric(encoded["is_weekend"], errors="coerce").fillna(0).astype(int)

    for column, mapping in CATEGORY_MAPS.items():
        default_value = DEFAULT_CATEGORY_VALUES[column]
        if column not in encoded.columns:
            encoded[column] = next(iter(mapping.keys()))
        normalized = encoded[column].astype(str).str.lower()
        encoded[column] = normalized.map(mapping).fillna(default_value).astype(int)

    return encoded[MODEL_FEATURES]
