import os
from fastapi import FastAPI
import httpx

SERVICE_NAME = os.getenv("SERVICE_NAME", "service_a")
PEER_URL = os.getenv("PEER_URL", "http://service_b:8001")

app = FastAPI(title="Service A")

@app.get("/ping")
def ping():
    return {"service": SERVICE_NAME, "status": "ok"}

@app.post("/ack")
async def ack(payload: dict):
    return {"service": SERVICE_NAME, "received": payload}

@app.post("/send")
async def send(payload: dict):
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(f"{PEER_URL}/process", json={"from": SERVICE_NAME, "data": payload})
        return {"service": SERVICE_NAME, "peer_response": r.json()}

@app.get("/demo")
async def demo():
    payload = {"message": "hello from A"}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(f"{PEER_URL}/process", json={"from": SERVICE_NAME, "data": payload})
        return {"service": SERVICE_NAME, "sent": payload, "peer_response": r.json()}