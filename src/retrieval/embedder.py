from typing import List
from src.services.ollama_service import OllamaService
import logging

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """
    Interface for generating embedding vectors for user queries through an external 
    embedding service, producing vector representations suitable for similarity search.

    Attributes:
        ollama (OllamaService): Service instance responsible for computing embedding vectors.

    Methods:
        embed(text: str) -> List[float]:
            Generates an embedding vector for the provided query text using the configured embedding service.
    """

    def __init__(self, ollama: OllamaService) -> None:
        self.ollama = ollama

    async def embed(self, text: str) -> List[float]:
        logger.info("Starting query embedding generation")
        vector = await self.ollama.embed(text)
        logger.info("Query embedding generation completed")
        return vector