import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    ollama_chat_model: str = field(default_factory=lambda: os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"))

    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "127.0.0.1"))
    qdrant_port: int = field(default_factory=lambda: _get_int("QDRANT_PORT", 6333))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "documents"))

    chunk_size_words: int = field(default_factory=lambda: _get_int("CHUNK_SIZE_WORDS", 480))
    chunk_overlap_words: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP_WORDS", 75))
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _get_float("SIMILARITY_THRESHOLD", 0.65))
    max_history_turns: int = field(default_factory=lambda: _get_int("MAX_HISTORY_TURNS", 6))

    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "./data/uploads"))
    cors_origins: list[str] = field(default_factory=lambda: _get_list("CORS_ORIGINS", "http://localhost:5173"))


settings = Settings()
