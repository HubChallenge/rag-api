from fastapi import APIRouter, HTTPException

from app.config import settings
from app.services.ollama_models import list_chat_models

router = APIRouter()


@router.get("")
def get_models() -> dict:
    try:
        models = list_chat_models(settings.ollama_base_url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama injoignable: {exc}") from exc

    return {"models": models, "default": settings.ollama_chat_model}
