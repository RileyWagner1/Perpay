# scoring.py
import pandas as pd  
import numpy as np
from datetime import datetime, timedelta
from config import config


def is_new_product(added_at):
    if pd.isna(added_at):  # type: ignore
        return False
    return (datetime.utcnow() - added_at.to_pydatetime()) <= timedelta(days=config.NEW_PRODUCT_DAYS)

def safe_get(row, key, default=0.0):
    val = row.get(key, default)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

def business_score(row, user_context) -> float:
    """
    row: dict-like row from catalog (already filtered)
    user_context: dict with flags like is_new_or_churn, abandoned_set, etc.
    """
    margin = safe_get(row, "margin_pct", 0.0) or safe_get(row, "margin_score", 0.0)
    conv_rate = safe_get(row, "conversion_rate", 0.0)
    refund_p = safe_get(row, "refund_probability", 0.0)
    added_at = row.get("product_added_date")
    new_bonus = 1.0 if is_new_product(added_at) else 0.0

    abandoned_bonus = 1.0 if row.get("product_id") in user_context.get("abandoned_set", set()) else 0.0

    score  = config.W_MARGIN * margin
    score += (config.W_CONV * conv_rate) if user_context.get("is_new_or_churn", False) else 0.0
    score += config.W_RECENCY_NEW * new_bonus
    score += config.W_ABANDONED_CART * abandoned_bonus
    score -= config.W_REFUND_RISK * refund_p
    return float(score)

def final_score(user_sim: float, business: float) -> float:
    return config.W_USER_SIM * user_sim + business
