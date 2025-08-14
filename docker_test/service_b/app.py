import os
from fastapi import FastAPI
import httpx

SERVICE_NAME = os.getenv("SERVICE_NAME", "service_b")
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://service_a:8000/ack")

app = FastAPI(title="Service B")

@app.get("/ping")
def ping():
    return {"service": SERVICE_NAME, "status": "ok"}

@app.post("/process")
async def process(payload: dict):
    processed = {"processed_by": SERVICE_NAME, "received": payload}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            cb = await client.post(CALLBACK_URL, json=processed)
            cb_json = cb.json()
    except Exception as e:
        cb_json = {"error": str(e)}
    return {"service": SERVICE_NAME, "status": "processed", "callback_result": cb_json}