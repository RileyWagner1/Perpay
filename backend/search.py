# backend/search.py
from __future__ import annotations
import re, math
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Cleaning helpers ----------
EXCEL_ERR = re.compile(r"^\s*#(?:REF|NAME|VALUE|NULL|N/?A|DIV/0!?|NUM|CALC)!?\s*$", re.I)
def _clean_token_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if EXCEL_ERR.match(s) else s

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\-\/\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _plural_to_singular(q: str) -> str:
    toks, out = (q or "").split(), []
    for t in toks:
        if len(t) > 3 and t.endswith("es") and not t.endswith("ses"): out.append(t[:-2])
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"): out.append(t[:-1])
        else: out.append(t)
    return " ".join(out)

def _minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)

# --------- Core class ----------
class CosineSearch:
    """
    TF-IDF search with field weighting + optional business scoring.
    Final score = alpha * cosine + (1 - alpha) * business_score
    where business_score is a weighted mix of normalized business features.
    """

    # Map of business features -> dataframe columns (or derived)
    DEFAULT_FEATURE_MAP = {
        "profit_margin": ("_derived_margin", None),     # (price - cost) / price
        "gross_margin": ("current_margin", None),       # if present
        "conversion": ("conversion_rate", None),
        "inventory": ("current_inventory", None),
        "price": ("current_price", "inverse"),          # lower price = better -> inverse after scaling
        "ship_price": ("current_ship_price", "inverse")
    }

    def __init__(self, csv_path: str, *, name_col: str = "name"):
        self.name_col = name_col
        self.df = pd.read_csv(csv_path)

        # IDs
        if "product_id" in self.df.columns:
            self.df["product_id"] = (
                self.df["product_id"].astype(str).str.replace(r"[,\s]", "", regex=True).str.strip()
            )

        # Clean text fields
        for col in [self.name_col, "brand", "category_name_1", "category_name_2", "category_name_3"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].map(_clean_token_text)

        # Derived margin if possible
        if "current_price" in self.df.columns and "current_cost" in self.df.columns:
            pr = pd.to_numeric(self.df["current_price"], errors="coerce")
            ct = pd.to_numeric(self.df["current_cost"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                self.df["_derived_margin"] = np.where(pr > 0, (pr - ct) / pr, np.nan)

        # === Build weighted TF-IDF ===
        name  = self.df[self.name_col].fillna("").astype(str).map(_norm)
        brand = self.df.get("brand", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm)
        c1 = self.df.get("category_name_1", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm)
        c2 = self.df.get("category_name_2", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm)
        c3 = self.df.get("category_name_3", pd.Series("", index=self.df.index)).fillna("").astype(str).map(_norm)

        self.df["search_name"]  = (name + " ").str.strip()
        self.df["search_brand"] = (brand + " ").str.strip()
        self.df["search_cat"]   = (c1 + " " + c2 + " " + c3).str.strip()

        self.v_name = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode",
                                      ngram_range=(1, 3), sublinear_tf=True, smooth_idf=True,
                                      min_df=2, max_df=0.6)
        self.v_brand = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode",
                                       ngram_range=(1, 2), sublinear_tf=True, smooth_idf=True,
                                       min_df=2, max_df=0.8)
        self.v_cat = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode",
                                     ngram_range=(1, 2), sublinear_tf=True, smooth_idf=True,
                                     min_df=2, max_df=0.9)

        Xn = self.v_name.fit_transform(self.df["search_name"])
        Xb = self.v_brand.fit_transform(self.df["search_brand"])
        Xc = self.v_cat.fit_transform(self.df["search_cat"])

        # weights: name 3x, brand 2x, cats 1x
        self.X = hstack([3 * Xn, 2 * Xb, 1 * Xc]).tocsr()

        # Precompute normalized business features (0..1)
        self.biz_feature_names, self.biz_matrix = self._build_business_matrix()

    # ---------- Business features ----------
    def _build_business_matrix(self) -> Tuple[List[str], np.ndarray]:
        cols, mats = [], []
        for feat, (col, mode) in self.DEFAULT_FEATURE_MAP.items():
            if col not in self.df.columns:
                continue
            s = pd.to_numeric(self.df[col], errors="coerce")
            z = _minmax(s.fillna(s.median()))
            if mode == "inverse":
                z = 1.0 - z  # cheaper = better
            cols.append(feat)
            mats.append(z.values.reshape(-1, 1))
        if not mats:
            return [], np.zeros((len(self.df), 0))
        M = np.hstack(mats)  # N x F
        return cols, M

    # ---------- Encoders ----------
    def _encode_query(self, q: str):
        q = _norm(_plural_to_singular(q or ""))
        qn = self.v_name.transform([q])
        qb = self.v_brand.transform([q])
        qc = self.v_cat.transform([q])
        return hstack([3 * qn, 2 * qb, 1 * qc]).tocsr()

    # ---------- Main search ----------
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_cols: Optional[List[str]] = None,
        candidates_idx: Optional[np.ndarray] = None,
        alpha: float = 0.7,                    # blend: 0..1  (1 = cosine only)
        biz_weights: Optional[Dict[str, float]] = None,  # e.g., {"profit_margin": 0.5, "conversion": 0.3, "inventory": 0.2}
    ) -> List[dict]:
        if not query or not query.strip():
            return []

        q_vec = self._encode_query(query)

        # Cosine similarity
        if candidates_idx is None:
            sims = cosine_similarity(q_vec, self.X).ravel()
            idx_all = np.arange(self.X.shape[0])
        else:
            sims = cosine_similarity(q_vec, self.X[candidates_idx]).ravel()
            idx_all = candidates_idx

        # Business score
        if self.biz_matrix.size and (biz_weights is not None) and len(biz_weights) > 0:
            w = np.array([biz_weights.get(f, 0.0) for f in self.biz_feature_names], dtype=float)  # F
            if np.allclose(w.sum(), 0.0):
                biz = np.zeros_like(sims)
            else:
                w = w / (abs(w).sum())  # L1 normalize so magnitudes are stable
                biz_full = self.biz_matrix @ w  # N
                biz = biz_full[idx_all]
        else:
            biz = np.zeros_like(sims)

        # Blend
        alpha = float(np.clip(alpha, 0.0, 1.0))
        score = alpha * sims + (1.0 - alpha) * biz

        order = np.argsort(-score)[: int(top_k)]
        idx = idx_all[order]

        out = self.df.iloc[idx].copy()
        out.insert(0, "similarity", np.round(sims[order], 4))
        out.insert(1, "business_score", np.round(biz[order], 4))
        out.insert(2, "score", np.round(score[order], 4))   # final blended score (used for ranking)

        if include_cols is None:
            include_cols = ["score", "similarity", "business_score", "product_id",
                            self.name_col, "brand", "current_price", "product_url"]
        include_cols = [c for c in include_cols if c in out.columns]

        result = out[include_cols].to_dict(orient="records")
        def _to_native(v):
            return v.item() if isinstance(v, np.generic) else v
        for r in result:
            for k, v in list(r.items()):
                r[k] = _to_native(v)
        return result
