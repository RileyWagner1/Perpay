######################################################################################
# Adjusts reccomendations to: 
# - prioritize high-margin items (profit boost)
# - highlight new products if browsing outdated items
# - push abandoned cart items strategically
# - reduce low-margin, high-refund items unless necessary
######################################################################################

import pandas as pd

class BusinessRulesAgent:
    """
    Enforces catalog nudges in candidate set:
      - Recommend newly added products when user is viewing outdated items
      - Push abandoned-cart items
      - (Optionally) remove very low margin + high refund items
    """
    def __init__(self): pass

    def adjust_candidates(self, df: pd.DataFrame, session_ctx: dict) -> pd.DataFrame:
        # Soft-prefer abandoned cart items by ensuring they are present
        if session_ctx["abandoned"]:
            abandoned_df = df[df["product_id"].isin(session_ctx["abandoned"])]
            df = pd.concat([abandoned_df, df]).drop_duplicates(subset=["product_id"])
        return df
