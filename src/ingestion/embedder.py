import logging
from typing import List, Dict, Any
from uuid import uuid4
from src.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


class Embedder:
    """
    Interface for generating embedding vectors from textual chunks using an external 
    embedding service and producing structured vector‑payload artifacts for storage.

    Attributes:
        ollama (OllamaService): Service instance used to generate embedding vectors for input chunks.

    Methods:
        embed_chunks(chunks: List[str]) -> List[Dict[str, Any]]:
            Generates embedding vectors for all provided text chunks and returns structured 
            vector records containing identifiers, vectors, and associated payloads.
    """

    def __init__(self, ollama_service: OllamaService) -> None:
        self.ollama = ollama_service
        logger.info("Embedder initialized with OllamaService instance.")

    async def embed_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        if not chunks:
            raise ValueError("Chunk list is empty.")

        logger.info("Starting embedding generation for all chunks.")

        results: List[Dict[str, Any]] = []
        for chunk in chunks:
            vector = await self.ollama.embed(chunk)
            results.append(
                {
                    "id": str(uuid4()),
                    "vector": vector,
                    "payload": {"text": chunk},
                }
            )

        logger.info(f"Embedding generation completed for {len(results)} chunks.")
        return results