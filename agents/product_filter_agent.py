######################################################################################
# This filters the catalog based on explicit user constraints (price, category, brand)
######################################################################################
# agents/product_filter_agent.py
from typing import Dict, Any
import pandas as pd

class ProductFilterAgent:
    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog

    def filter(self, constraints: Dict[str, Any]) -> pd.DataFrame:
        df = self.catalog.copy()

        if "category" in constraints:
            cat = constraints["category"].lower()
            cand = []
            for c in ["category_name_1","category_name_2","category_name_3"]:
                if c in df.columns:
                    cand.append(df[df[c].astype(str).str.lower().str.contains(cat, na=False)])
            if cand:
                df = pd.concat(cand).drop_duplicates(subset=["product_id"])

        if "brand" in constraints and "brand" in df.columns:
            df = df[df["brand"].astype(str).str.lower() == str(constraints["brand"]).lower()]

        # Only apply price filters if price column exists
        if "price" in df.columns:
            if "max_price" in constraints:
                df = df[pd.to_numeric(df["price"], errors="coerce") <= float(constraints["max_price"])]
            if "min_price" in constraints:
                df = df[pd.to_numeric(df["price"], errors="coerce") >= float(constraints["min_price"])]

        if "exclude_product_ids" in constraints:
            df = df[~df["product_id"].isin(set(constraints["exclude_product_ids"]))]

        return df.head(1000)
