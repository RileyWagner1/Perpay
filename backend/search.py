# backend/search.py
from __future__ import annotations

import json
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _norm_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["name", "brand", "category_name_1", "category_name_2", "category_name_3"] if c in df.columns]
    if "name" not in cols:
        raise ValueError("Expected a 'name' column in CSV.")
    tmp = df[cols].fillna("").astype(str)
    text = tmp.apply(lambda r: " ".join([_norm_str(x) for x in r]), axis=1)
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


class CosineSearch:
    """
    Minimal TF-IDF + cosine search over product catalog.
    """

    def __init__(self, csv_path: str, *, name_col: str = "name"):
        self.name_col = name_col
        self.df = pd.read_csv(csv_path)
        if self.name_col not in self.df.columns:
            raise ValueError(f"CSV missing required column '{self.name_col}'")

        # Build search corpus
        self.corpus_df = self.df.copy()
        self.corpus_df = self.corpus_df[~self.corpus_df[self.name_col].isna()].copy()
        self.corpus_df["search_text"] = _build_search_text(self.corpus_df)
        self.corpus_df = self.corpus_df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            strip_accents="unicode",
            ngram_range=(1, 3),
        )
        self.X = self.vectorizer.fit_transform(self.corpus_df["search_text"])

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_cols: Optional[list[str]] = None,
    ) -> List[dict]:
        if not query or not query.strip():
            return []

        q_vec = self.vectorizer.transform([query.strip().lower()])
        sims = cosine_similarity(q_vec, self.X).ravel()
        idx = np.argsort(-sims)[: int(top_k)]

        out = self.corpus_df.iloc[idx].copy()
        out.insert(0, "similarity", np.round(sims[idx], 4))

        # Default columns to return
        if include_cols is None:
            include_cols = [
                "similarity",
                "product_id",
                self.name_col,
                "brand",
                "current_price",
                "product_url",
            ]
        include_cols = [c for c in include_cols if c in out.columns]

        # Convert to JSON-serializable dicts
        result = out[include_cols].to_dict(orient="records")

        # Ensure numpy scalars -> native
        def _to_native(v):
            return v.item() if isinstance(v, np.generic) else v

        for r in result:
            for k, v in list(r.items()):
                r[k] = _to_native(v)

        return result
