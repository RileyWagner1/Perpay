import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class CosineSearch:
    def __init__(self, csv_path: str, name_col: str | None = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        self.df = pd.read_csv(csv_path)
        if name_col and name_col in self.df.columns:
            self.name_col = name_col
        else:
            for c in ["name", "product_name", "title"]:
                if c in self.df.columns:
                    self.name_col = c
                    break
            else:
                text_cols = [c for c in self.df.columns if self.df[c].dtype == "object"]
                self.name_col = text_cols[0] if text_cols else self.df.columns[0]
        self.df[self.name_col] = self.df[self.name_col].fillna("").astype(str)
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", strip_accents="unicode", ngram_range=(1,3))
        self.X = self.vectorizer.fit_transform(self.df[self.name_col])

    def search(self, query: str, top_k: int = 5):
        if not query: return []
        q_vec = self.vectorizer.transform([str(query)])
        sims = linear_kernel(q_vec, self.X).ravel()
        idx = np.argsort(-sims)[:top_k]
        out = self.df.iloc[idx].copy()
        out.insert(0, "similarity", np.round(sims[idx], 4))
        return out.to_dict(orient="records")
