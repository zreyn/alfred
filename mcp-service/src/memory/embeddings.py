"""
Ollama embeddings client for generating text embeddings.
"""

import logging
from typing import Optional

import httpx

from memory.config import OLLAMA_HOST, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Async client for generating embeddings via Ollama API."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = EMBEDDING_MODEL,
        timeout: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()
        url = f"{self.host}/api/embed"

        try:
            response = await client.post(
                url,
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns embeddings in the "embeddings" field
            embeddings = data.get("embeddings", [])

            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama embed API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
