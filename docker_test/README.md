# Two Python containers (A <-> B) — local Docker

Two FastAPI services exchange JSON. A sends to B; B calls back to A.

## Quick start
```bash
docker compose up --build -d
curl http://localhost:8000/ping
curl http://localhost:8001/ping
```

## Round trip demo (A -> B -> A)
```bash
curl -X POST http://localhost:8000/send       -H "Content-Type: application/json"       -d '{"order_id": 123, "note": "round trip"}'
```

## Endpoints
- A: GET /ping, POST /send, POST /ack
- B: GET /ping, POST /process