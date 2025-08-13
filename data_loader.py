# data_loader.py
import numpy as np
import pandas as pd
from typing import Tuple
from config import config

# ---------- helpers ----------
def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _impute_price(df: pd.DataFrame) -> pd.Series:
    # start empty float series
    p = pd.Series(np.nan, index=df.index, dtype="float64")

    # 1) current_price (your dataset says 0 missing)
    if "current_price" in df.columns:
        p = p.fillna(_coerce_float(df["current_price"]))

    # 2) msrp (~90%) as fallback
    if "msrp" in df.columns:
        msrp = _coerce_float(df["msrp"])
        p = p.fillna((msrp * 0.90).round(2))

    # 3) revenue/demand heuristic
    for rev_col, dem_col in [("revenue_90_days","demand_90_days"), ("revenue_30_days","demand_30_days")]:
        if rev_col in df.columns and dem_col in df.columns:
            rev = _coerce_float(df[rev_col])
            dem = _coerce_float(df[dem_col]).replace({0.0: np.nan})
            est = rev / dem
            p = p.fillna(est)

    # 4) brand/vendor medians on whatever we have so far
    if "brand" in df.columns:
        p = p.fillna(p.groupby(df["brand"]).transform("median"))
    if "current_vendor" in df.columns:
        p = p.fillna(p.groupby(df["current_vendor"]).transform("median"))

    # 5) global fallback
    global_guess = float(p.dropna().median()) if p.notna().any() else 49.99
    p = p.fillna(global_guess)

    return p.clip(lower=5.0, upper=1999.0).round(2)

def _impute_margin_pct(df: pd.DataFrame) -> pd.Series:
    m = _coerce_float(df["margin_pct"]) if "margin_pct" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")

    # category priors (tweak as you like)
    cat_map = {
        "Electronics": 0.12, "Auto": 0.15, "Auto Accessories": 0.18,
        "Home": 0.25, "Outdoor Living & Garden": 0.22, "Outdoor": 0.22,
        "Beauty": 0.40, "Baby & Kids": 0.28, "Furniture": 0.30,
        "Appliances": 0.18, "Sports": 0.22
    }
    cat1 = df.get("category_name_1")
    base = cat1.map(lambda c: cat_map.get(str(c), 0.22)) if cat1 is not None else pd.Series(0.22, index=df.index)
    m = m.fillna(base)

    # only fill from existing margin column if present
    if "current_vendor" in df.columns and "margin_pct" in df.columns:
        m = m.fillna(df.groupby("current_vendor")["margin_pct"].transform("median"))
    if "brand" in df.columns and "margin_pct" in df.columns:
        m = m.fillna(df.groupby("brand")["margin_pct"].transform("median"))

    return m.clip(lower=0.05, upper=0.80).round(4)

def _impute_conversion_rate(df: pd.DataFrame) -> pd.Series:
    cr = _coerce_float(df["conversion_rate"]) if "conversion_rate" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")

    # rank inversion (higher rank -> higher score)
    for rank_col in ["catalog_demand_30_days_rank", "catalog_demand_90_days_rank",
                     "catalog_revenue_30_days_rank", "catalog_revenue_90_days_rank"]:
        if rank_col in df.columns:
            r = _coerce_float(df[rank_col])
            max_r = float(r.max()) if r.notna().any() else None
            if max_r and max_r > 0:
                score = 1.0 - (r / max_r)
                est = (0.01 + 0.19 * score).round(4)  # ~1%..20%
                cr = cr.fillna(est)

    # carts/demand proxy if available
    for cart_col, dem_col in [("distinct_users_parent_carted_30","demand_30_days"),
                              ("distinct_users_parent_carted_60","demand_90_days"),
                              ("distinct_users_parent_carted_90","demand_90_days")]:
        if cart_col in df.columns and dem_col in df.columns:
            carts = _coerce_float(df[cart_col])
            dem = _coerce_float(df[dem_col]).replace({0.0: np.nan})
            est = (carts / dem).clip(0.001, 0.5)
            cr = cr.fillna(est)

    cr = cr.fillna(0.03)
    return cr.clip(lower=0.001, upper=0.6).round(4)

def _impute_refund_probability(df: pd.DataFrame) -> pd.Series:
    rp = _coerce_float(df["refund_probability"]) if "refund_probability" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")

    cat_priors = {
        "Electronics": 0.08, "Auto": 0.03, "Home": 0.05, "Beauty": 0.04,
        "Baby & Kids": 0.06, "Furniture": 0.09, "Appliances": 0.07, "Sports": 0.05
    }
    cat1 = df.get("category_name_1")
    base = cat1.map(lambda c: cat_priors.get(str(c), 0.06)) if cat1 is not None else pd.Series(0.06, index=df.index)
    rp = rp.fillna(base)

    # small uplift for high priced items
    price = _coerce_float(df["price"]) if "price" in df.columns else pd.Series(np.nan, index=df.index)
    rp = rp + (price > 500).astype(float) * 0.01 + (price > 1000).astype(float) * 0.01

    return rp.clip(lower=0.0, upper=0.5).round(4)

def _enrich_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "product_added_date" in out.columns:
        out["product_added_date"] = pd.to_datetime(out["product_added_date"], errors="coerce")

    out["price"] = _impute_price(out)
    out["margin_pct"] = _impute_margin_pct(out)
    out["conversion_rate"] = _impute_conversion_rate(out)
    out["refund_probability"] = _impute_refund_probability(out)
    return out

# ---------- public API ----------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(config.CATALOG_PATH)
    orders = pd.read_csv(config.ORDER_HISTORY_PATH)

    for c in ["order_carted_date","order_checkout_date","order_denied_date","order_approved_date",
              "order_in_repayment_date","order_refunded_date","order_canceled_date","order_complete_date"]:
        if c in orders.columns:
            orders[c] = pd.to_datetime(orders[c], errors="coerce")

    catalog = _enrich_catalog(catalog)
    return catalog, orders

__all__ = ["load_data"]
