from pathlib import Path

from app.services.chunker import chunk_sections
from app.services.embeddings import OllamaEmbeddings
from app.services.parsers import get_file_type, parse_file
from app.services.qdrant_store import QdrantStore

BATCH_SIZE = 8


class EmptyDocumentError(Exception):
    pass


def ingest_file(
    path: Path,
    source: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    embeddings: OllamaEmbeddings,
    store: QdrantStore,
) -> tuple[str, int]:
    file_type = get_file_type(path)
    sections = parse_file(path)
    chunks = chunk_sections(sections, chunk_size_words, chunk_overlap_words)

    if not chunks:
        raise EmptyDocumentError(f"No extractable text found in '{source}'")

    store.ensure_collection(embeddings.get_dimension())
    store.delete_by_source(source)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embeddings.embed_document_chunks([chunk.text for chunk in batch])
        store.upsert_chunks(source, file_type, batch, vectors)

    return file_type, len(chunks)
