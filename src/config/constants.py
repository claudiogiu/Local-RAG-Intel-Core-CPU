import os
from typing import Optional


OLLAMA_BASE_URL: Optional[str] = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL: Optional[str] = os.getenv("OLLAMA_EMBED_MODEL")
OLLAMA_CHAT_MODEL: Optional[str] = os.getenv("OLLAMA_CHAT_MODEL")

QDRANT_BASE_URL: Optional[str] = os.getenv("QDRANT_BASE_URL")
QDRANT_COLLECTION_NAME: Optional[str] = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_VECTOR_SIZE: Optional[int] = (
    int(os.getenv("QDRANT_VECTOR_SIZE"))
    if os.getenv("QDRANT_VECTOR_SIZE") is not None
    else None
)

HF_DATASET_NAME: Optional[str] = os.getenv("HF_DATASET_NAME")
HF_DATASET_SUBSET: Optional[str] = os.getenv("HF_DATASET_SUBSET")