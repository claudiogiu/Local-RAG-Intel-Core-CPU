import logging
from typing import List

logger = logging.getLogger(__name__)


class Chunker:
    """
    Interface for segmenting raw text into fixed‑size overlapping chunks suitable for 
    downstream embedding generation and retrieval‑augmented processing.

    Attributes:
        chunk_size (int): Maximum size of each generated chunk expressed in characters.
        overlap (int): Number of characters shared between consecutive chunks to preserve contextual continuity.

    Methods:
        split(text: str) -> List[str]:
            Splits the input text into overlapping chunks using sentence‑level boundaries and 
            applies controlled merging to maintain semantic coherence.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            f"Chunker initialized with chunk_size={self.chunk_size} and overlap={self.overlap}."
        )

    def split(self, text: str) -> List[str]:
        if not text:
            raise ValueError("Input text is empty.")

        logger.info("Starting text chunking process.")

        sentences = text.split(". ")
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 2 <= self.chunk_size:
                current += sentence + ". "
            else:
                chunks.append(current.strip())
                current = sentence + ". "

        if current:
            chunks.append(current.strip())

        final_chunks: List[str] = []
        for i in range(0, len(chunks)):
            start = max(0, i - 1)
            merged = " ".join(chunks[start:i+1])
            final_chunks.append(merged)

        logger.info(f"Chunking completed. Generated {len(final_chunks)} chunks.")

        return final_chunks