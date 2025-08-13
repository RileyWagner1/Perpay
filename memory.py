from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

class InMemorySessionStore:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "viewed": [],
            "cart": [],
            "abandoned_cart": [],
            "messages": [],
            "created_at": datetime.utcnow(),
            "user_id": None
        })

    def init(self, session_id: str, user_id: int | None = None):
        _ = self.sessions[session_id]
        if user_id is not None:
            self.sessions[session_id]["user_id"] = user_id

    def get(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]

    def add_viewed(self, session_id: str, product_id: int):
        s = self.sessions[session_id]
        if product_id not in s["viewed"]:
            s["viewed"].append(product_id)

    def add_cart(self, session_id: str, product_id: int):
        s = self.sessions[session_id]
        if product_id not in s["cart"]:
            s["cart"].append(product_id)
        # remove from abandoned if re-carted
        if product_id in s["abandoned_cart"]:
            s["abandoned_cart"].remove(product_id)

    def abandon_cart(self, session_id: str, product_id: int):
        s = self.sessions[session_id]
        if product_id in s["cart"] and product_id not in s["abandoned_cart"]:
            s["abandoned_cart"].append(product_id)

store = InMemorySessionStore()
