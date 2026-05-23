import logging
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import (
    RAGIngestResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGRetrieveRequest,
    RAGRetrieveResponse,
    RAGEmbedRequest,
    RAGEmbedResponse,
)
from src.api.fields import RetrievalStatus, PipelineStatus
from src.core.orchestrator import get_orchestrator
from src.ingestion.loader import Loader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.uploader import Uploader
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/ingest", response_model=RAGIngestResponse, tags=["RAG"])
async def rag_ingest(request: Request):
    """
    Execute the ingestion pipeline by loading documents, splitting them into chunks,
    generating embeddings, and uploading vectors to the vector store.
    """

    try:
        logger.info("Starting ingestion process")

        loader = Loader()
        chunker = Chunker()

        orchestrator = get_orchestrator()
        embedder = Embedder(orchestrator.ollama)
        uploader = Uploader(orchestrator.qdrant)

        documents: List[str] = loader.load()
        logger.info(f"Loaded {len(documents)} documents.")

        chunks: List[str] = []
        for doc in documents:
            chunks.extend(chunker.split(doc))
        logger.info(f"Generated {len(chunks)} chunks.")

        embeddings = await embedder.embed_chunks(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings.")

        await uploader.upload(embeddings)
        logger.info("Upload to Qdrant completed.")

        return RAGIngestResponse(
            status=PipelineStatus.COMPLETED,
            documents=len(documents),
            chunks=len(chunks),
            embeddings=len(embeddings)
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")


@router.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(request: Request, payload: RAGQueryRequest):
    """
    Perform a full RAG query by retrieving relevant documents and generating
    an LLM-based answer using the retrieved context.
    """

    try:
        orchestrator = get_orchestrator()
        answer, retrieved = await orchestrator.query(payload.query, payload.limit)

        return RAGQueryResponse(
            status=PipelineStatus.COMPLETED,
            answer=answer,
            context=[doc["payload"]["text"] for doc in retrieved],
            retrieved=[
                {
                    "id": doc["id"],
                    "score": doc["score"],
                    "text": doc["payload"]["text"]
                }
                for doc in retrieved
            ]
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed")


@router.post("/rag/retrieve", response_model=RAGRetrieveResponse, tags=["RAG"])
async def rag_retrieve(request: Request, payload: RAGRetrieveRequest):
    """
    Retrieve the most relevant documents from the vector store based on
    semantic similarity to the input query.
    """

    try:
        orchestrator = get_orchestrator()
        results = await orchestrator.retrieve(payload.query, payload.limit)

        return RAGRetrieveResponse(
            status=RetrievalStatus.SUCCESS,
            retrieved=[
                {
                    "id": doc["id"],
                    "score": doc["score"],
                    "text": doc["payload"]["text"]
                }
                for doc in results
            ]
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed")


@router.post("/rag/embed", response_model=RAGEmbedResponse, tags=["RAG"])
async def rag_embed(request: Request, payload: RAGEmbedRequest):
    """
    Generate an embedding vector for the provided text using the designated
    embedding model.
    """

    try:
        orchestrator = get_orchestrator()
        vector = await orchestrator.embed(payload.text)
        return RAGEmbedResponse(embedding=vector)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed")