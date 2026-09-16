from app.config import settings
from app.services.embeddings import OllamaEmbeddings
from app.services.qdrant_store import QdrantStore

embeddings = OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.ollama_embed_model)
store = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection=settings.qdrant_collection)
