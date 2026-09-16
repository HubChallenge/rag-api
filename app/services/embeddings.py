import ollama


class OllamaEmbeddings:
    def __init__(self, base_url: str, model: str) -> None:
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._dimension: int | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embed(model=self._model, input=texts)
        return [list(vector) for vector in response.embeddings]

    def embed_document_chunks(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {text}" for text in texts]
        return self.embed_texts(prefixed)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([f"search_query: {text}"])
        return vectors[0]

    def get_dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_texts(["search_document: dimension probe"])
            self._dimension = len(probe[0])
        return self._dimension
