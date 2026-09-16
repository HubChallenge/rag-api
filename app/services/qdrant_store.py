import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.services.chunker import Chunk


def _point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_index}"))


class QdrantStore:
    def __init__(self, host: str, port: int, collection: str) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    def ping(self) -> bool:
        self._client.get_collections()
        return True

    def ensure_collection(self, vector_size: int) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        source: str,
        file_type: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        indexed_at = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=_point_id(source, chunk.chunk_index),
                vector=vector,
                payload={
                    "source": source,
                    "file_type": file_type,
                    "page": chunk.page,
                    "heading": chunk.heading,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "collection": self._collection,
                    "indexed_at": indexed_at,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def _source_filter(self, source: str) -> Filter:
        return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])

    def delete_by_source(self, source: str) -> None:
        if not self._collection_exists():
            return
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(filter=self._source_filter(source)),
        )

    def count_by_source(self, source: str) -> int:
        if not self._collection_exists():
            return 0
        result = self._client.count(
            collection_name=self._collection,
            count_filter=self._source_filter(source),
            exact=True,
        )
        return result.count

    def search(self, query_vector: list[float], top_k: int):
        if not self._collection_exists():
            return []
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
        )
        return response.points

    def list_sources(self) -> list[dict]:
        if not self._collection_exists():
            return []

        aggregates: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                source = payload.get("source")
                if not source:
                    continue
                entry = aggregates.setdefault(
                    source,
                    {
                        "source": source,
                        "file_type": payload.get("file_type", ""),
                        "chunk_count": 0,
                        "indexed_at": payload.get("indexed_at", ""),
                    },
                )
                entry["chunk_count"] += 1
                indexed_at = payload.get("indexed_at", "")
                if indexed_at and indexed_at < entry["indexed_at"]:
                    entry["indexed_at"] = indexed_at
            if offset is None:
                break

        return list(aggregates.values())

    def _collection_exists(self) -> bool:
        existing = [c.name for c in self._client.get_collections().collections]
        return self._collection in existing
