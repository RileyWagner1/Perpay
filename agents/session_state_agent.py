######################################################################################
# This tracks viewed product, carted products, abandonded products, current conversation goals
######################################################################################


from typing import Dict, Any, List
from memory import store

class SessionStateAgent:
    def __init__(self, mem=store):
        self.mem = mem

    def get_context(self, session_id: str) -> Dict[str, Any]:
        s = self.mem.get(session_id)
        return {
            "viewed": s["viewed"],
            "cart": s["cart"],
            "abandoned": s["abandoned_cart"],
            "user_id": s["user_id"],
        }

    def mark_viewed(self, session_id: str, product_id: int): self.mem.add_viewed(session_id, product_id)
    def add_to_cart(self, session_id: str, product_id: int): self.mem.add_cart(session_id, product_id)
    def mark_abandoned(self, session_id: str, product_id: int): self.mem.abandon_cart(session_id, product_id)
