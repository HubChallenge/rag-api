import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import settings
from app.deps import embeddings, store
from app.schemas import AskRequest, SourceRef
from app.services.llm import stream_answer

router = APIRouter()


def _build_sources(results: list) -> list[SourceRef]:
    return [
        SourceRef(
            source=(result.payload or {}).get("source", ""),
            page=(result.payload or {}).get("page"),
            chunk_index=(result.payload or {}).get("chunk_index", 0),
            score=result.score,
        )
        for result in results
    ]


@router.post("/ask")
def ask_question(request: AskRequest) -> StreamingResponse:
    top_k = request.top_k or settings.top_k

    query_vector = embeddings.embed_query(request.question)
    all_results = store.search(query_vector, top_k)
    relevant_results = [r for r in all_results if r.score >= settings.similarity_threshold]
    sources = _build_sources(relevant_results)

    history = request.history[-(settings.max_history_turns * 2) :]

    def event_stream() -> Iterator[str]:
        yield json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]}) + "\n"
        try:
            for event in stream_answer(
                question=request.question,
                results=relevant_results,
                history=history,
                base_url=settings.ollama_base_url,
                model=settings.ollama_chat_model,
            ):
                if event["event"] == "thinking":
                    yield json.dumps({"type": "thinking"}) + "\n"
                elif event["event"] == "token":
                    yield json.dumps({"type": "token", "content": event["text"]}) + "\n"
                elif event["event"] == "done":
                    yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
