from typing import Literal

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    source: str
    file_type: str
    chunk_count: int
    indexed_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class UploadResponse(BaseModel):
    source: str
    file_type: str
    chunks_indexed: int
    status: str


class DeleteResponse(BaseModel):
    source: str
    deleted: bool
    chunks_deleted: int


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = []
    top_k: int | None = None


class SourceRef(BaseModel):
    source: str
    page: int | None
    chunk_index: int
    score: float


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    qdrant: bool
