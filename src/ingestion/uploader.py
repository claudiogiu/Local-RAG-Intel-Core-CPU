import logging
from typing import List, Dict, Any
from src.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class Uploader:
    """
    Interface for managing batched insertion of vector records into a Qdrant collection, 
    ensuring controlled upload throughput and reliable persistence of embedding artifacts.

    Attributes:
        qdrant (QdrantService): Service instance responsible for executing upsert operations in Qdrant.
        batch_size (int): Fixed number of vector records processed per upload batch.

    Methods:
        upload(points: List[Dict[str, Any]]) -> None:
            Uploads vector records to Qdrant in fixed‑size batches, creating the collection 
            if necessary and ensuring sequential persistence of all provided points.
    """

    def __init__(self, qdrant_service: QdrantService) -> None:
        self.qdrant = qdrant_service
        self.batch_size = 1000
        logger.info("Uploader initialized with QdrantService instance.")

    async def upload(self, points: List[Dict[str, Any]]) -> None:
        if not points:
            raise ValueError("Point list is empty.")

        logger.info("Starting upload of vector points to Qdrant.")

        await self.qdrant.create_collection_if_not_exists()

        total = len(points)
        logger.info(f"Uploading {total} points in batches of {self.batch_size}.")

        for start in range(0, total, self.batch_size):
            end = start + self.batch_size
            batch = points[start:end]

            logger.info(f"Uploading batch {start} to {end} ({len(batch)} points).")

            await self.qdrant.upsert_points(batch)

        logger.info("Upload to Qdrant completed.")
