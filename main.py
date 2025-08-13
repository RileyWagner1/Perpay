######################################################################################
# we'll use RAG layer: uses vector search over: 
# - product descriptions + category tags
# - user purchase history embeddings
# - can be implemented with FAISS locally for hackathon speed
######################################################################################


# MVP Data Flow Overview: 
# 1. User sends a message --> orchestrator parses intent + constraints
# 2. Orchestrator queries Session State Agent and User Profile Agent 
# 3. Product Filter Agent narrows cataglog (from CSV)
# 4. Reccomendation Ranking Agent scores remaining products based on: 
# - user preference mathc
# - business priority score 

# 5. Orchestrator returns top N suggestions w/explainations










# Business Goal: for scoring, keep it simple --> these weights can be tuned to max profits later (or find which is most important)
# def business_score(product, user): 
#     score = 0
#     score += .3 * product["margin_score"]
#     score += .25 * product["conversion_rate"] if user.is_new_or_churn else 0
#     score += .2 * (1 if product["is_new"] else 0)
#     score += .15 * (1 if product["in_abandoned_cart"] else 0)
#     score -= .1 * product["refund_probability"]
#     return score





import argparse
from data_loader import load_data
from orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str, default="demo-session")
    parser.add_argument("--user_id", type=int, default=0)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--brand", type=str, default=None)
    parser.add_argument("--min_price", type=float, default=None)
    parser.add_argument("--max_price", type=float, default=None)
    args = parser.parse_args()

    catalog, orders = load_data()
    o = Orchestrator(catalog, orders)

    query = {k: v for k, v in vars(args).items() if k in {"category","brand","min_price","max_price"} and v is not None}
    recs = o.recommend(args.session_id, args.user_id, query)

    try:
        from ui.simple_cli import pretty
        print(pretty(recs).to_string(index=False))
    except Exception:
        print(recs.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
