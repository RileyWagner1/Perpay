import json
import uuid
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

from data_loader import load_data
from orchestrator import Orchestrator
from models.llm_client import OllamaClient

# ----------------------------
# App bootstrap
# ----------------------------
st.set_page_config(page_title="Perpay AI Shopping Assistant (MVP)", layout="wide")

# One-time init
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = 0  # anonymous by default

# Load data / orchestrator (cache to keep things snappy)
@st.cache_resource(show_spinner=False)
def _get_orchestrator():
    catalog, orders = load_data()
    return Orchestrator(catalog, orders), catalog, orders

orchestrator, catalog_df, orders_df = _get_orchestrator()

@st.cache_resource(show_spinner=False)
def _get_llm():
    return OllamaClient(model="qwen3:0.6b")

llm = _get_llm()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Session & Filters")
st.sidebar.caption("You can set a demo user_id or keep anonymous (0).")
st.session_state.user_id = st.sidebar.number_input("user_id", value=st.session_state.user_id, step=1, min_value=0)

with st.sidebar.expander("Optional explicit filters", expanded=False):
    explicit_category = st.text_input("category (e.g., TV, Auto, Home)", value="")
    explicit_brand = st.text_input("brand (e.g., Samsung, Sony)", value="")
    col_min, col_max = st.columns(2)
    with col_min:
        explicit_min_price = st.text_input("min_price", value="")
    with col_max:
        explicit_max_price = st.text_input("max_price", value="")

if st.sidebar.button("Clear chat & viewed"):
    st.session_state.messages = []
    # basic clear: new session id to reset viewed/cart in our simple in-memory store
    st.session_state.session_id = f"sess-{uuid.uuid4().hex[:8]}"
    st.rerun()

# ----------------------------
# Helper: extract constraints via LLM
# ----------------------------
EXTRACT_PROMPT = """System: You are a helpful shopping assistant that extracts simple filters from user requests.
User will ask for products. Output ONLY a compact JSON with keys among:
  "category" (string), "brand" (string), "min_price" (number), "max_price" (number).
If a key isn't present in the request, omit it. No extra text, only JSON.

Examples:
USER: I'm looking for a Samsung TV under $500
JSON: {"category":"TV","brand":"Samsung","max_price":500}

USER: Need a lawn mower, any brand, $200-$400
JSON: {"category":"Lawn Mower","min_price":200,"max_price":400}

Now process the next USER message.
"""

def llm_extract_constraints(user_text: str) -> Dict[str, Any]:
    try:
        raw = llm.chat([
            {"role":"system","content":"You extract shopping filters in JSON."},
            {"role":"user","content":EXTRACT_PROMPT + "\nUSER: " + user_text}
        ])
        # Try to locate a JSON object in the response
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start:end+1])
            # Normalize keys
            norm = {}
            if "category" in obj and isinstance(obj["category"], str) and obj["category"].strip():
                norm["category"] = obj["category"].strip()
            if "brand" in obj and isinstance(obj["brand"], str) and obj["brand"].strip():
                norm["brand"] = obj["brand"].strip()
            if "min_price" in obj:
                try: norm["min_price"] = float(obj["min_price"])
                except: pass
            if "max_price" in obj:
                try: norm["max_price"] = float(obj["max_price"])
                except: pass
            return norm
    except Exception as e:
        # For MVP, just swallow and return empty constraints
        pass
    return {}

# ----------------------------
# Main pane: Chat UI
# ----------------------------
st.title("🛒 Perpay AI Shopping Assistant (MVP)")
st.caption("Chat to discover products. The app uses an orchestrator with business-aware ranking and simple session memory.")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_text = st.chat_input("Describe what you're shopping for (e.g., 'Looking for a dash cam under $150')")

def merge_constraints(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge explicit sidebar filters on top of LLM-parsed constraints (explicit wins).
    """
    final = dict(parsed)
    if explicit_category.strip(): final["category"] = explicit_category.strip()
    if explicit_brand.strip(): final["brand"] = explicit_brand.strip()
    if explicit_min_price.strip():
        try: final["min_price"] = float(explicit_min_price)
        except: pass
    if explicit_max_price.strip():
        try: final["max_price"] = float(explicit_max_price)
        except: pass
    return final

if user_text:
    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 1) Parse constraints with LLM
    with st.spinner("Thinking..."):
        parsed = llm_extract_constraints(user_text)
        constraints = merge_constraints(parsed)

    # 2) Call orchestrator for recommendations
    with st.spinner("Finding options..."):
        df = orchestrator.recommend(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            query=constraints
        )

    # 3) Show results as simple product cards
    with st.chat_message("assistant"):
        st.markdown("Here are some picks I think you'll like:")
        if df.empty:
            st.info("No results. Try broadening your filters.")
        else:
            # Minimal columns to display
            show_cols = ["product_id","name","brand","category_name_1","category_name_2","product_added_date"]
            show_cols = [c for c in show_cols if c in df.columns]
            for _, row in df.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.markdown(f"**{row.get('name','(no name)')}**")
                        meta = []
                        if "brand" in row and pd.notna(row["brand"]): meta.append(f"Brand: {row['brand']}")
                        cat1 = row.get("category_name_1")
                        if cat1 and pd.notna(cat1): meta.append(f"Category: {cat1}")
                        cat2 = row.get("category_name_2")
                        if cat2 and pd.notna(cat2): meta.append(f"Subcategory: {cat2}")
                        if row.get("__score__") is not None:
                            meta.append(f"Score: {row['__score__']:.3f}")
                        st.caption(" • ".join(meta) if meta else " ")

                    with col2:
                        pid = int(row["product_id"])
                        add = st.button("Add to cart", key=f"add_{pid}")
                        abandon = st.button("Abandon", key=f"ab_{pid}")
                        view = st.button("View", key=f"vw_{pid}")

                        if add:
                            orchestrator.session.add_to_cart(st.session_state.session_id, pid)
                            st.success("Added to cart")
                        if abandon:
                            orchestrator.session.mark_abandoned(st.session_state.session_id, pid)
                            st.warning("Marked abandoned")
                        if view:
                            orchestrator.session.mark_viewed(st.session_state.session_id, pid)
                            st.info("Marked viewed")

        # Keep the assistant reply in chat history (optional summary)
        if not df.empty:
            names = ", ".join(map(str, df["name"].head(5).tolist()))
            st.session_state.messages.append({
                "role":"assistant",
                "content": f"I found {len(df)} options. Top picks: {names}."
            })
        else:
            st.session_state.messages.append({
                "role":"assistant",
                "content": "I couldn't find any products with those filters. Try relaxing price/brand/category."
            })

# ----------------------------
# Footer: Session snapshots
# ----------------------------
with st.expander("Session snapshot"):
    ctx = orchestrator.session.get_context(st.session_state.session_id)
    st.write({"session_id": st.session_state.session_id, **ctx})

with st.expander("Cart"):
    ctx = orchestrator.session.get_context(st.session_state.session_id)
    cart_ids = set(ctx.get("cart", []))
    if not cart_ids:
        st.caption("Cart is empty.")
    else:
        cart_df = catalog_df[catalog_df["product_id"].isin(cart_ids)]
        st.dataframe(cart_df[["product_id","name","brand"]], hide_index=True)

