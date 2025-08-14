# backend/app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from search import CosineSearch

CSV_PATH = os.getenv("CSV_PATH", "/data/product_catalog.csv")
NAME_COL = os.getenv("NAME_COL", "name")

app = FastAPI(title="Cosine Similarity Backend")

# CORS so Streamlit (different origin) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you have a fixed frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
startup_error = ""
try:
    engine = CosineSearch(CSV_PATH, name_col=NAME_COL)
except Exception as e:
    startup_error = str(e)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    items: List[Dict[str, Any]]

@app.get("/healthz")
def healthz():
    if engine is None:
        return {"status": "degraded", "error": startup_error}
    return {"status": "ok", "rows": len(engine.df), "name_col": engine.name_col}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")
    items = engine.search(req.query, top_k=req.top_k)
    return {"items": items}
