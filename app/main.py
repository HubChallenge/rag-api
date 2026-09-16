from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.deps import embeddings, store
from app.routers import chat, documents
from app.schemas import HealthResponse

app = FastAPI(title="RAG Local API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.on_event("startup")
def on_startup() -> None:
    try:
        store.ensure_collection(embeddings.get_dimension())
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = False
    qdrant_ok = False

    try:
        embeddings.embed_texts(["ping"])
        ollama_ok = True
    except Exception:
        ollama_ok = False

    try:
        store.ping()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return HealthResponse(status="ok", ollama=ollama_ok, qdrant=qdrant_ok)
