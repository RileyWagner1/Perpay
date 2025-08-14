# frontend/app.py
import os
import requests
import pandas as pd
import streamlit as st
from llm_client import extract_keywords, summarize_products

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="LLM + Cosine Search", page_icon="🛍️", layout="centered")
st.title("🛍️ LLM-powered Product Finder")
st.caption("Type what you're looking for. We’ll condense your query, search the catalog, and summarize matches.")

with st.form(key="query_form"):
    user_text = st.text_input("What are you looking for?", placeholder="e.g., i am looking for expensive laptops")
    submitted = st.form_submit_button("Search")

if submitted and user_text.strip():
    with st.spinner("Calling LLM to extract keywords..."):
        keywords = extract_keywords(user_text)
    st.write(f"**Extracted keywords:** `{keywords}`")

    with st.spinner("Querying cosine-similarity backend..."):
        items = []
        try:
            r = requests.post(f"{BACKEND_URL}/search", json={"query": keywords, "top_k": 5}, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
        except Exception as e:
            st.error(f"Backend error: {e}")

    if items:
        df = pd.DataFrame(items)
        st.dataframe(df, use_container_width=True)
        with st.spinner("Calling LLM to write a readable summary..."):
            summary = summarize_products(user_text, keywords, items)
        st.markdown("---")
        st.markdown("### Summary")
        st.write(summary)
    else:
        st.info("No results found. Try different wording.")
else:
    st.write("Enter a product request above and press **Search**.")
