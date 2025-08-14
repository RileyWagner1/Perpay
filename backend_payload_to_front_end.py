
from cosine_similarity_test import get_top_similar_items
from data_preprocess_solved import products_df, practice_JSON

# will need to import functions from business_side.py once we get the return values for the "best products"

# format this into a JSON file such that it represents this structure: 
# {prod_id_1: {"name": "product name", "price": "current_price", 
# prod_id_2: {...}, ...}


# create function to take the top n items from dataframe and return JSON file format



import json
import numpy as np
import pandas as pd
from typing import Iterable, Mapping, Optional, Union

def topn_df_to_json_map(
    df: pd.DataFrame,
    n: int = 10,
    *,
    key_col: str = "product_id",
    include: Iterable[str] = ("name", "current_price"),
    rename: Optional[Mapping[str, str]] = {"current_price": "price"},
    as_string: bool = False,
    write_path: Optional[str] = None,
) -> Union[dict, str]:
    """
    Build {product_id: {...}} JSON-like mapping from top-n rows of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Your topN result (must contain `key_col`).
    n : int
        Number of items to include.
    key_col : str
        Column to use as the JSON map key (e.g., 'product_id').
    include : Iterable[str]
        Columns to include as values for each product.
    rename : Mapping[str, str] or None
        Optional column rename mapping (e.g., {'current_price': 'price'}).
    as_string : bool
        If True, return a pretty-printed JSON string; else return a dict.
    write_path : str or None
        If provided, write JSON to this path.

    Returns
    -------
    dict or str
        Dict (by default) or JSON string if `as_string=True`.
    """
    # Ensure columns exist
    cols = [key_col] + [c for c in include if c in df.columns]
    if key_col not in df.columns:
        raise ValueError(f"Missing key_col '{key_col}' in DataFrame.")
    if len(cols) == 1:
        raise ValueError("No value columns found from `include` in DataFrame.")

    sub = df[cols].head(n).copy()

    # Optional rename (e.g., current_price -> price)
    if rename:
        sub = sub.rename(columns=rename)

    # Replace NaN with None and convert numpy scalars to native Python types
    sub = sub.where(pd.notna(sub), None)

    def _to_native(v):
        return v.item() if isinstance(v, np.generic) else v

    sub = sub.applymap(_to_native)

    # Build {product_id: {...}} map
    out = {
        str(row[key_col]): {k: row[k] for k in sub.columns if k != key_col}
        for _, row in sub.iterrows()
    }

    # Optionally write to file / return string
    if write_path:
        with open(write_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if as_string:
        return json.dumps(out, ensure_ascii=False, indent=2)

    return out





# Or as a JSON string:
topN = topN, extras = get_top_similar_items(products_df, practice_JSON, top_k=200, anchor_drop=False)
json_str = topn_df_to_json_map(topN, n=5, as_string=True)
print(json_str)