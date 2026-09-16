from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.deps import embeddings, store
from app.schemas import DeleteResponse, DocumentInfo, DocumentListResponse, UploadResponse
from app.services.ingestion import EmptyDocumentError, ingest_file
from app.services.parsers import UnsupportedFileType

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / filename
    dest_path.write_bytes(file.file.read())

    try:
        file_type, chunk_count = ingest_file(
            path=dest_path,
            source=filename,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            embeddings=embeddings,
            store=store,
        )
    except UnsupportedFileType as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(
        source=filename,
        file_type=file_type,
        chunks_indexed=chunk_count,
        status="success",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = [DocumentInfo(**doc) for doc in store.list_sources()]
    return DocumentListResponse(documents=documents)


@router.delete("/{source}", response_model=DeleteResponse)
def delete_document(source: str) -> DeleteResponse:
    chunk_count = store.count_by_source(source)
    if chunk_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{source}' not found")

    store.delete_by_source(source)

    file_path = Path(settings.upload_dir) / source
    if file_path.exists():
        file_path.unlink()

    return DeleteResponse(source=source, deleted=True, chunks_deleted=chunk_count)
