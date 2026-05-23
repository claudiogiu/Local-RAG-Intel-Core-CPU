from typing import List, Dict, Any
from src.retrieval.embedder import QueryEmbedder
from src.services.qdrant_service import QdrantService
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """
    Interface for executing semantic retrieval operations by embedding user queries 
    and performing vector‑based similarity search over a Qdrant collection.

    Attributes:
        query_embedder (QueryEmbedder): Component responsible for generating embedding vectors for input queries.
        qdrant (QdrantService): Service instance used to perform vector search operations in Qdrant.

    Methods:
        retrieve(query: str, limit: int = 5) -> List[Dict[str, Any]]:
            Generates an embedding for the input query, executes a vector similarity search, 
            and returns the top matching results with associated metadata.
    """

    def __init__(self, query_embedder: QueryEmbedder, qdrant: QdrantService) -> None:
        self.query_embedder = query_embedder
        self.qdrant = qdrant

    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        logger.info("Starting retrieval process")
        vector = await self.query_embedder.embed(query)
        logger.info("Query embedding generated, starting vector search")

        results = self.qdrant.search(query_vector=vector, limit=limit)

        logger.info(f"Vector search completed, retrieved {len(results)} results")

        for idx, item in enumerate(results):
            point_id = item.get("id")
            score = item.get("score")
            payload_keys = list(item.get("payload", {}).keys())
            logger.info(
                f"Result {idx}: id={point_id}, score={score}, payload_keys={payload_keys}"
            )

        logger.info("Retrieval process completed")
        return results