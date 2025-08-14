# backend/app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from search import CosineSearch

CSV_PATH = os.getenv("CSV_PATH", "/data/product_catalog.csv")
NAME_COL = os.getenv("NAME_COL", "name")

app = FastAPI(title="Cosine Similarity Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="blend between cosine and business score")
    biz_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="e.g. {'profit_margin':0.5,'conversion':0.3,'inventory':0.2,'price':0.0}"
    )

class SearchResponse(BaseModel):
    items: List[Dict[str, Any]]

@app.get("/healthz")
def healthz():
    if engine is None:
        return {"status": "degraded", "error": startup_error}
    return {
        "status": "ok",
        "rows": len(engine.df),
        "name_col": engine.name_col,
        "biz_features": engine.biz_feature_names,
        "alpha_default": 0.7,
    }

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")
    items = engine.search(
        req.query,
        top_k=req.top_k,
        alpha=req.alpha,
        biz_weights=req.biz_weights,
    )
    return {"items": items}
