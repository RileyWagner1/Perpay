from dataclasses import dataclass

@dataclass
class Config:
    CATALOG_PATH: str = "data/product_catalog.csv"
    ORDER_HISTORY_PATH: str = "data/user_order_history.csv"

    # Business weights (hackathon-tunable)
    W_USER_SIM: float = 0.35
    W_MARGIN: float = 0.25
    W_CONV: float = 0.20
    W_RECENCY_NEW: float = 0.10
    W_ABANDONED_CART: float = 0.07
    W_REFUND_RISK: float = 0.12  # subtract

    NEW_PRODUCT_DAYS: int = 45   # what counts as "new"
    CHURN_INACTIVE_DAYS: int = 45

config = Config()
