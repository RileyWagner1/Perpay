# agents/user_profile_agent.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

class UserProfileAgent:
    def __init__(self, orders: pd.DataFrame, catalog: pd.DataFrame):
        self.orders = orders
        self.catalog = catalog.copy()   # <- remove set_index(..., drop=False)

    def build_profile(self, user_id: int) -> Dict[str, Any]:
        u = self.orders[self.orders["user_id"] == user_id]
        profile = {"fav_brands": {}, "fav_categories": {}, "price_range": (None, None), "is_new_or_churn": False}

        if u.empty:
            profile["is_new_or_churn"] = True
            return profile

        # Select only the columns that actually exist
        cols_needed = ["product_id","brand","category_name_1","category_name_2","category_name_3"]
        if "price" in self.catalog.columns:
            cols_needed.append("price")

        j = u.merge(self.catalog[cols_needed], on="product_id", how="left")

        # Favorites
        brand_counts = j["brand"].dropna().value_counts().to_dict() if "brand" in j.columns else {}
        cats = pd.concat(
            [j[c] for c in ["category_name_1","category_name_2","category_name_3"] if c in j.columns],
            axis=0
        ).dropna() if any(c in j.columns for c in ["category_name_1","category_name_2","category_name_3"]) else pd.Series(dtype=object)

        profile["fav_brands"] = brand_counts
        profile["fav_categories"] = cats.value_counts().to_dict() if not cats.empty else {}

        # Price range heuristic
        if "price" in j.columns and j["price"].notna().any():
            prices = pd.to_numeric(j["price"], errors="coerce").dropna()
            if not prices.empty:
                profile["price_range"] = (float(prices.quantile(0.25)), float(prices.quantile(0.75)))

        # Churn heuristic
        last_dt = u[["order_checkout_date","order_approved_date"]].max(axis=1).max()
        if pd.notna(last_dt):
            profile["is_new_or_churn"] = (datetime.utcnow() - last_dt.to_pydatetime()) > timedelta(days=45)
        else:
            profile["is_new_or_churn"] = True

        return profile

    def similarity(self, row, profile: Dict[str, Any]) -> float:
        s = 0.0
        if "brand" in row and row["brand"] and row["brand"] in profile["fav_brands"]:
            s += 1.0

        row_cats = {row.get("category_name_1"), row.get("category_name_2"), row.get("category_name_3")}
        row_cats = {c for c in row_cats if c}
        if row_cats:
            s += sum(profile["fav_categories"].get(c, 0) > 0 for c in row_cats) * 0.3

        pr = profile.get("price_range")
        if pr and pr[0] is not None and "price" in row and row["price"] is not None:
            lo, hi = pr
            try:
                p = float(row["price"])
                if lo <= p <= hi:
                    s += 0.5
            except Exception:
                pass
        return float(s)
