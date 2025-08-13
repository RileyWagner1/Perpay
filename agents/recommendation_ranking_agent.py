######################################################################################
# This ranks using weighted score
######################################################################################

# score = w1*user_similarity + w2*margin + w3*conversion_probability + w4*recency - w5*refund_risk

import pandas as pd
from typing import Dict, Any
from scoring import business_score, final_score

class RecommendationRankingAgent:
    """
    Combine user similarity + business objectives into final score.
    """
    def __init__(self, user_profile_agent):
        self.user_profile_agent = user_profile_agent

    def rank(self, df: pd.DataFrame, user_id: int, session_ctx: Dict[str, Any]) -> pd.DataFrame:
        profile = self.user_profile_agent.build_profile(user_id) if user_id else {"is_new_or_churn": True}

        context = {
            "is_new_or_churn": profile.get("is_new_or_churn", False),
            "abandoned_set": set(session_ctx.get("abandoned", []))
        }

        def _score_row(row):
            user_sim = self.user_profile_agent.similarity(row, profile)
            biz = business_score(row, context)
            return final_score(user_sim, biz)

        scored = df.copy()
        scored["__score__"] = scored.apply(_score_row, axis=1)
        return scored.sort_values("__score__", ascending=False)
