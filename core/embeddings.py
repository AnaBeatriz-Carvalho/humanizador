"""Embeddings locais via API compativel com OpenAI do LM Studio."""

from __future__ import annotations

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "text-embedding-nomic-embed-text-v1.5"


class EmbeddingClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self._is_available: bool | None = None

    def embed(self, texto: str) -> list[float]:
        return self.embed_many([texto])[0]

    def embed_many(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            return []
        response = self.client.embeddings.create(model=self.model, input=textos)
        data = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in data]

    def available(self) -> bool:
        """Testa uma vez se o servidor e o modelo de embeddings respondem."""
        if self._is_available is None:
            try:
                self.embed("teste")
                self._is_available = True
            except Exception:
                self._is_available = False
        return self._is_available
