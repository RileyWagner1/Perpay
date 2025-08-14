# frontend/llm_client.py
import os, re, json
from typing import List, Dict
from openai import OpenAI

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://ollama:11434/v1")
OLLAMA_KEY  = os.getenv("OLLAMA_KEY", "ollama")
MODEL       = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

client = OpenAI(base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)

def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    raw = m.group(0) if m else text
    try:
        return json.loads(raw)
    except Exception:
        raw = raw.replace("’","'").replace("“","\"").replace("”","\"")
        return json.loads(raw)


def _strip_think(text: str) -> str:
    return THINK_BLOCK.sub("", text or "").strip()

def extract_keywords(user_text: str, max_terms: int = 6) -> str:
    system = (
        "You are a shopping query condenser.\n"
        "Given a user message, output minimal search keywords.\n"
        f"Return ONLY JSON: {{\"keywords\": \"...\"}} with <= {max_terms} tokens.\n"
        "Lowercase; no punctuation besides hyphens; prefer singular nouns; keep brand/color/type/spec.\n"
        "Examples:\n"
        "  User: 'i want to see red shoe options, nike if possible' -> {\"keywords\": \"nike red shoe\"}\n"
        "  User: 'show samsung 55 inch 4k tvs' -> {\"keywords\": \"samsung 55-inch 4k tv\"}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_text}],
            response_format={"type":"json_object"},
            temperature=0.0,
        )
        content = resp.choices[0].message.content
    except Exception:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_text}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content

    data = _extract_json(content)
    kw = (data.get("keywords") or "").strip().lower()
    kw = re.sub(r"[^a-z0-9\-\s]", " ", kw)
    kw = re.sub(r"\s+", " ", kw).strip()
    seen, toks = set(), []
    for t in kw.split():
        if t not in seen:
            seen.add(t)
            toks.append(t)
    return " ".join(toks)

def summarize_products(original_query: str, keywords: str, items: List[Dict]) -> str:
    simple = []
    for it in items[:5]:
        simple.append({k: it.get(k) for k in ["similarity","name","brand","current_price","product_url","product_id"] if k in it})
    system = (
        "You write concise, human-friendly summaries of product search results.\n"
        "Use ONLY the provided products. Do not invent specs or prices.\n"
        "Output should be a short intro + a 5-item bulleted list (name, brand, price if present), then a one-line suggestion.\n"
    )
    user = json.dumps({"original_query": original_query, "keywords": keywords, "top5": simple}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.2,
        )
        return _strip_think(resp.choices[0].message.content.strip())
    except Exception:
        lines = [f"Top {len(simple)} matches for '{keywords}':"]
        for it in simple:
            nm = it.get("name") or "(name)"
            br = it.get("brand") or ""
            pr = it.get("current_price")
            line = f"- {nm}"
            if br: line += f" — {br}"
            if pr is not None: line += f" (${pr})"
            lines.append(line)
        return "\n".join(lines)
