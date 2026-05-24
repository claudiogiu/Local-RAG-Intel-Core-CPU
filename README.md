# Local RAG on Intel® Core™ CPUs via Ollama and Qdrant
## Introduction  

This repository is designed for deploying a fully local retrieval‑augmented generation system that operates on Intel® Core™ processors by adopting Ollama as the small‑language‑model runtime and Qdrant as the vector‑database component, integrating ingestion, embedding, storage, retrieval and generation within a unified environment that preserves data locality and supports controlled execution on CPU‑based hardware.

Ollama provides the local execution engine for the small language models used for embedding generation and text generation, while Qdrant offers the vector‑database capabilities required to store high‑dimensional embeddings and execute efficient similarity searches over the indexed corpus. 

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Configure the Environment File**  

- Initialize the environment configuration by copying the `.env.example` file template into the project root as `.env`:

  ```bash
  mv .env.example .env  
  ```

- Assign valid values to all required variables.

**2.** **Execute the Service Initialization with Makefile**  

- The repository includes a **Makefile** that automates the initialization of all components required to run the local RAG system.

- Run the following command to start the complete stack:

  ```bash
  make all
  ```

- This command sequentially performs the following operations:

  - Starts all required services using Docker Compose, ensuring that the API, the vector database and the model runtime are correctly instantiated.
  - Interprets the model identifiers provided through the environment variables `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL`, which respectively specify the embedding model used for vector generation and the generation model used for producing textual outputs.
  - Downloads the specified models from Ollama, ensuring that the embedding model is available for vector generation and that the generation model is available for text generation and RAG‑related operations.

**3.** **Access the API** 
  
  - Once the container is running, the API is accessible at:

    - **Swagger UI for interactive docs:** `localhost:8000/docs`  
    - **Document ingestion endpoint:** `api/v1/rag/ingest`  
    - **Retrieval‑augmented generation requests:** `api/v1/rag/query`
    - **Similarity‑based retrieval resource:** `api/v1/rag/retrieve`
    - **Embedding generation route:** `api/v1/rag/embed`  

## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
