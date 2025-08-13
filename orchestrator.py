from typing import Dict, Any
import pandas as pd
from agents.product_filter_agent import ProductFilterAgent
from agents.user_profile_agent import UserProfileAgent
from agents.session_state_agent import SessionStateAgent
from agents.business_rules_agent import BusinessRulesAgent
from agents.recommendation_ranking_agent import RecommendationRankingAgent

class Orchestrator:
    def __init__(self, catalog: pd.DataFrame, orders: pd.DataFrame):
        self.session = SessionStateAgent()
        self.filter = ProductFilterAgent(catalog)
        self.profile = UserProfileAgent(orders, catalog)
        self.rules = BusinessRulesAgent()
        self.rank = RecommendationRankingAgent(self.profile)
        self.catalog = catalog

    def recommend(self, session_id: str, user_id: int | None, query: Dict[str, Any]) -> pd.DataFrame:
        """
        query can contain: category, brand, min_price, max_price
        """
        # Session context
        ctx = self.session.get_context(session_id)
        if user_id is not None and ctx["user_id"] is None:
            from memory import store
            store.init(session_id, user_id)

        # 1) Filter
        candidates = self.filter.filter({
            **query,
            "exclude_product_ids": ctx["viewed"]  # example: avoid duplicating already surfaced items
        })

        # 2) Apply business policy adjustments to the candidate set
        candidates = self.rules.adjust_candidates(candidates, ctx)

        # 3) Rank
        uid = user_id if user_id is not None else 0
        ranked = self.rank.rank(candidates, uid, ctx)

        # 4) Persist “viewed” on top results
        top_ids = ranked["product_id"].head(10).tolist()
        for pid in top_ids:
            self.session.mark_viewed(session_id, int(pid))

        return ranked.head(10)
