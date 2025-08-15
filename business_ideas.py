# business_side.py
# REMOVE THIS COMMENT IF YOU HAVE THE FUNCTION DEFINED
from cosine_similarity_test import get_top_similar_items
# from data_preprocess_solved import products_df, practice_JSON

from config import config  # pulls weights W_MARGIN, W_USER_SIM
import pandas as pd
# sklearn normalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize(series: pd.Series) -> pd.Series:
    """ Min-max scale a pandas Series to [0,1], returning a new Series. """
    if series is None or series.empty:
        return series
    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    return pd.Series(scaled, index=series.index)

# weights - a * similarity + b * margin + c * recency - d * return_rate
# k - top 5
# pin_anchor - our anchor product (like main reference position or relevant product which will be compared with)

def rerank_by_business_score(topN: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder the topN DataFrame based on a simple business score:
        business_score = W_MARGIN * current_margin + W_USER_SIM * similarity

    Assumes that `current_margin` and `similarity` columns exist and are numeric.
    Returns the same DataFrame sorted by score (descending).
    """
    if topN is None or topN.empty:
        return topN

    df = topN.copy()

    # Ensure numeric
    df["current_margin"] = pd.to_numeric(df["current_margin"], errors="coerce").fillna(0.0)
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)

    # normalize
    df["current_margin"] = normalize(df["current_margin"])

    # add recency
    today = pd.to_datetime("today")
    df["product_added_date"] = pd.to_datetime(df["product_added_date"], errors="coerce")
    df["recency"] = (today - df["product_added_date"]).dt.days
    # normalize recency to [0, 1] by clipping to NEW_PRODUCT_DAYS
    df["recency"] = 1 - (df["recency"] / config.NEW_PRODUCT_DAYS)
    df["recency"] = df["recency"].clip(0, 1)
    
    # refund risk
    df["return_rate"] = pd.to_numeric(df["return_rate"], errors="coerce")

    # Compute score
    df["business_score"] = (
        config.W_MARGIN * df["current_margin"] +
        config.W_USER_SIM * df["similarity"] +
        config.W_RECENCY_NEW * df["recency"] - 
        config.W_REFUND_RISK * df["return_rate"]
    )

    # anchor-pin the original first row
    if len(df) > 1:
        anchor = df.iloc[[0]] 
        rest = df.iloc[1:].sort_values("business_score", ascending=False)
        # sort the rest by business_score
        df = pd.concat([anchor, rest], ignore_index=True)
    else:
        df = df.sort_values("business_score", ascending=False).reset_index(drop=True)

    return df

if __name__ == "__main__":

    show_cols = [
        "product_id", "name", "current_margin", "similarity", "recency", "return_rate", "business_score"
    ]
    # Example load (replace with get_top_similar_items output in real usage)
    topN = pd.read_csv("data/topN_results_with_returns.csv")
    print("Initial topN preview (top 10):")
    print(topN[show_cols[:len(show_cols) - 1]].head(10))

    reordered = rerank_by_business_score(topN)

    # Preview
    show_cols = [c for c in show_cols if c in reordered.columns]

    print("\nReordered preview (top 10):")
    print(reordered[show_cols].head(10))
# topN, extras = get_top_similar_items(products_df, practice_JSON, top_k=200, return_intermediates=True)
# print(topN.head(10))