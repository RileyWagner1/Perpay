# cosine_similarity_test.py
# pip install pandas scikit-learn numpy

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------- Helpers -----------------------

def _norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()

def apply_price_filter(df: pd.DataFrame, min_price, max_price, price_col: str = "current_price") -> pd.DataFrame:
    if price_col not in df.columns:
        return df
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    if min_price is not None:
        out = out[out[price_col] >= float(min_price)]
    if max_price is not None:
        out = out[out[price_col] <= float(max_price)]
    return out

def filter_d1(df: pd.DataFrame, payload: Dict) -> pd.DataFrame:
    out = df.copy()
    for level in ["category_name_1", "category_name_2", "category_name_3"]:
        if level in out.columns and payload.get(level):
            val = str(payload[level]).strip().lower()
            out = out[out[level].astype(str).str.strip().str.lower() == val]
    if payload.get("brand") and "brand" in out.columns:
        b = str(payload["brand"]).strip().lower()
        out = out[out["brand"].astype(str).str.strip().str.lower() == b]
    out = apply_price_filter(out, payload.get("min_price"), payload.get("max_price"), price_col="current_price")
    return out

def build_search_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["name", "brand", "category_name_1", "category_name_2", "category_name_3"] if c in df.columns]
    if "name" not in cols:
        raise ValueError("Expected a 'name' column in product_catalog.csv")
    tmp = df[cols].fillna("").astype(str)
    text = tmp.apply(lambda r: " ".join([_norm_str(x) for x in r]), axis=1)
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

def build_query_text(payload: Dict) -> str:
    bits = []
    for k in ["category_name_1", "category_name_2", "category_name_3", "brand", "object", "description"]:
        v = payload.get(k)
        if v:
            bits.append(str(v))
    if payload.get("min_price") is not None:
        bits.append(f"minprice {payload['min_price']}")
    if payload.get("max_price") is not None:
        bits.append(f"maxprice {payload['max_price']}")
    return " ".join(bits).lower().strip()

def cosine_name_search(vectorizer, X, corpus_df: pd.DataFrame, query: str, top_k: int = 10, display_cols=None) -> pd.DataFrame:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).ravel()
    idx = np.argsort(-sims)[:top_k]
    out = corpus_df.iloc[idx].copy()
    out.insert(0, "similarity", np.round(sims[idx], 4))
    if display_cols is None:
        display_cols = ["similarity", "name"]
        for c in ["product_id", "brand", "current_price", "product_url"]:
            if c in out.columns:
                display_cols.append(c)
    return out[display_cols]


# ----------------------- Public API -----------------------

def get_top_similar_items(
    products_df: pd.DataFrame,
    payload: Dict,
    *,
    top_k: int = 200,
    price_col: str = "current_price",
    return_intermediates: bool = False,
    anchor_drop: bool = True,   # <-- NEW: set to False for query-by-example UIs
) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
    """
    Compute top-k products most similar (cosine over TF-IDF of name/brand/categories)
    to the centroid of the anchor set defined by `payload`.

    Parameters
    ----------
    products_df : pd.DataFrame
        Catalog with at least a 'name' column; optionally 'brand', category_name_1..3, product_id, current_price.
    payload : dict
        Filter/query payload (e.g., category_name_*, brand, min_price, max_price, object, description).
    top_k : int, default 200
        Number of similar items to return.
    price_col : str, default "current_price"
        Price column name (if present).
    return_intermediates : bool, default False
        If True, also return intermediates (d1, d2, corpus_df, vectorizer, X).
    anchor_drop : bool, default True
        If True, remove anchors from results. Set to False for query-by-example (keep anchors).

    Returns
    -------
    topN : pd.DataFrame
        DataFrame of top-k similar items with a 'similarity' column.
    extras : dict or None
        If `return_intermediates=True`, returns a dict with d1, d2, corpus_df, vectorizer, X.
    """
    # Build TF-IDF corpus
    corpus_df = products_df.copy()
    corpus_df = corpus_df[~corpus_df["name"].isna()].copy()
    corpus_df["search_text"] = build_search_text(corpus_df)
    corpus_df = corpus_df.reset_index(drop=True)  # align indices with X rows

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        strip_accents="unicode",
        ngram_range=(1, 3),
    )
    X = vectorizer.fit_transform(corpus_df["search_text"])

    # Anchor & candidates
    d1 = filter_d1(products_df, payload)
    if d1.empty:
        qtext = build_query_text(payload)
        proxy = cosine_name_search(vectorizer, X, corpus_df, qtext, top_k=1)
        if proxy.empty:
            raise ValueError("No candidates found at all.")
        if "product_id" in proxy.columns and "product_id" in products_df.columns:
            pid = proxy["product_id"].iloc[0]
            d1 = products_df[products_df["product_id"] == pid]
        else:
            pname = proxy["name"].iloc[0]
            d1 = products_df[products_df["name"] == pname]

    d2 = apply_price_filter(products_df, payload.get("min_price"), payload.get("max_price"), price_col=price_col)

    # Map to TF-IDF rows
    if "product_id" in products_df.columns and "product_id" in corpus_df.columns:
        d1_ids = set(d1["product_id"].tolist())
        d2_ids = set(d2["product_id"].tolist())
        d1_corpus = corpus_df[corpus_df["product_id"].isin(d1_ids)]
        d2_corpus = corpus_df[corpus_df["product_id"].isin(d2_ids)]
    else:
        d1_names = set(d1["name"].tolist())
        d2_names = set(d2["name"].tolist())
        d1_corpus = corpus_df[corpus_df["name"].isin(d1_names)]
        d2_corpus = corpus_df[corpus_df["name"].isin(d2_names)]

    if d1_corpus.empty or d2_corpus.empty:
        raise ValueError("No overlap between filtered sets and TF-IDF corpus.")

    # Centroid cosine similarity
    d1_idx = d1_corpus.index.to_list()
    d2_idx = d2_corpus.index.to_list()

    centroid = np.asarray(X[d1_idx].mean(axis=0)).reshape(1, -1)
    sims = cosine_similarity(centroid, X[d2_idx]).ravel()

    d2_corpus = d2_corpus.copy()
    d2_corpus["similarity"] = sims

    # Conditionally exclude anchors
    if anchor_drop:
        if "product_id" in d1_corpus.columns and "product_id" in d2_corpus.columns:
            exclude_ids = set(d1_corpus["product_id"].tolist())
            d2_corpus = d2_corpus[~d2_corpus["product_id"].isin(exclude_ids)]
        else:
            exclude_names = set(d1_corpus["name"].tolist())
            d2_corpus = d2_corpus[~d2_corpus["name"].isin(exclude_names)]

    topN = d2_corpus.sort_values("similarity", ascending=False).head(top_k)

    extras = None
    if return_intermediates:
        extras = {
            "d1": d1,
            "d2": d2,
            "corpus_df": corpus_df,
            "vectorizer": vectorizer,
            "X": X,
        }
    return topN, extras



