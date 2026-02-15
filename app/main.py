from fastapi import FastAPI

from app.api import ingest, query

app = FastAPI(title="Multimodal RAG", version="0.1.0")

app.include_router(ingest.router, tags=["ingest"])
app.include_router(query.router, tags=["query"])


@app.get("/health")
async def health():
    return {"status": "ok"}
