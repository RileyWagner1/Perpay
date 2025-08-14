# data_preprocess.py
import pandas as pd

products_df = pd.read_csv("product_catalog.csv")
user_orders_df = pd.read_csv("user_order_history.csv")  # not used yet

practice_JSON = {
    "session_id": "sess-123",
    "user_id": 0,
    "category_name_1": "Auto",
    "category_name_2": "Audio/Video",
    "category_name_3": "Dash Cams",
    "brand": "Adesso",
    "min_price": 80,
    "max_price": 150,
    "object": "dash cam",
    "description": "cheap, reliable, good night vision"
}
