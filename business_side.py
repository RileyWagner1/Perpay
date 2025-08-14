# another_script.py
from cosine_similarity_test import get_top_similar_items
from data_preprocess_solved import products_df, practice_JSON

topN, extras = get_top_similar_items(products_df, practice_JSON, top_k=200, return_intermediates=True)
print(topN.head(10))