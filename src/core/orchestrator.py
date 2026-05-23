import logging
from typing import List, Dict, Tuple
from src.retrieval.embedder import QueryEmbedder
from src.retrieval.retriever import Retriever
from src.services.ollama_service import OllamaService
from src.services.qdrant_service import QdrantService
from src.prompts.rag_instruction import RAG_INSTRUCTION


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Interface for coordinating embedding, retrieval, and generation components into a unified 
    retrieval‑augmented generation (RAG) system.

    Attributes:
        ollama (OllamaService): Service instance responsible for embedding and LLM text generation.
        qdrant (QdrantService): Service instance managing vector storage and similarity search.
        embedder (QueryEmbedder): Component used to generate embedding vectors for input text.
        retriever (Retriever): Component executing semantic retrieval over the vector database.

    Methods:
        query(text: str, limit: int = 5) -> Tuple[str, List[Dict]]:
            Executes a full RAG pipeline by retrieving contextual documents, constructing 
            an augmented prompt, and generating an LLM response.

        retrieve(text: str, limit: int = 5) -> List[Dict]:
            Performs semantic retrieval only, returning the top matching vector records.

        embed(text: str) -> List[float]:
            Generates an embedding vector for the provided text using the embedding service.

        shutdown() -> None:
            Gracefully terminates active services and releases associated resources.
    """

    def __init__(self):
        logger.info("Initializing Orchestrator")
        self.ollama = OllamaService()
        self.qdrant = QdrantService()
        self.embedder = QueryEmbedder(self.ollama)
        self.retriever = Retriever(self.embedder, self.qdrant)
        logger.info("Orchestrator initialized successfully")

    async def query(self, text: str, limit: int = 5) -> Tuple[str, List[Dict]]:
        logger.info("Executing RAG query")
        retrieved = await self.retriever.retrieve(text, limit)
        context = "\n\n".join(doc["payload"]["text"] for doc in retrieved)
        prompt = RAG_INSTRUCTION.format(context=context, query=text)
        answer = await self.ollama.chat(prompt)
        return answer, retrieved

    async def retrieve(self, text: str, limit: int = 5) -> List[Dict]:
        logger.info("Executing retrieval operation")
        return await self.retriever.retrieve(text, limit)

    async def embed(self, text: str) -> List[float]:
        logger.info("Generating embedding vector")
        return await self.embedder.embed(text)

    async def shutdown(self) -> None:
        logger.info("Shutting down Orchestrator services")
        await self.qdrant.close()
        logger.info("Orchestrator shutdown completed")


_orchestrator_instance: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance