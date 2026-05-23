import logging
import httpx
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
from src.config.constants import (
    QDRANT_BASE_URL,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Interface for managing vector storage, collection administration, and similarity search 
    operations within a Qdrant vector database instance.

    Attributes:
        base_url (str): Base URL of the Qdrant service used for all API interactions.
        collection_name (str): Name of the target collection where vectors and payloads are stored.
        vector_size (int): Dimensionality of the embedding vectors expected by the collection.
        _client (QdrantClient): Synchronous Qdrant client used for collection management and upsert operations.

    Methods:
        _collection_exists() -> bool:
            Checks whether the configured collection is already present in the Qdrant instance.

        create_collection_if_not_exists() -> None:
            Creates the configured collection if it does not already exist, applying the defined vector parameters.

        upsert_points(points: List[Dict[str, Any]]) -> None:
            Inserts or updates vector points and associated payloads into the configured collection.

        search(query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
            Executes a similarity search against the configured collection and returns the top matching results.

        close() -> None:
            Closes the Qdrant service and releases associated resources.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        self.base_url = base_url or QDRANT_BASE_URL
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or QDRANT_VECTOR_SIZE

        self._client = QdrantClient(url=self.base_url)

        logger.info(
            f"Qdrant service initialization completed. "
            f"Base endpoint: {self.base_url}. "
            f"Collection: {self.collection_name}. "
            f"Vector size: {self.vector_size}."
        )

    async def _collection_exists(self) -> bool:
        logger.info("Checking whether the collection exists in Qdrant.")
        collections = self._client.get_collections()
        names = [c.name for c in collections.collections]
        return self.collection_name in names

    async def create_collection_if_not_exists(self) -> None:
        exists = await self._collection_exists()

        if exists:
            logger.info("The collection already exists in Qdrant.")
            return

        logger.info("Creating the collection in Qdrant.")
        self._client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info("Collection successfully created.")

    async def upsert_points(
        self,
        points: List[Dict[str, Any]]
    ) -> None:
        logger.info("Upserting points into the Qdrant collection.")

        qdrant_points = []
        for p in points:
            qdrant_points.append(
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {})
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )

        logger.info("Point upsert operation completed.")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        logger.info("Vector search in Qdrant initiated.")

        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }

        response = httpx.post(
            f"{self.base_url}/collections/{self.collection_name}/points/search?with_payload=*&with_vector=false",
            json=payload
        )

        response.raise_for_status()
        data = response.json()

        logger.info("Vector search completed.")
        return data.get("result", [])

    async def close(self) -> None:
        logger.info("Qdrant service closed.")