Deployable RAG Service for ChatGPT Conversation Restoration

A Retrieval-Augmented Generation (RAG) system that restores and queries long ChatGPT conversation histories using semantic search and a local LLM backend.
The project demonstrates end-to-end AI system design, from offline data ingestion and vector indexing to an API-driven, Dockerized inference service.

✨ What this project does

Ingests ChatGPT conversations.json exports

Builds a FAISS vector index over dialogue-level chat content

Retrieves relevant historical chat segments for a given query

Generates grounded responses using a local LLM (via Ollama)

Exposes the full pipeline as a FastAPI service

Packages the system into a reproducible Docker deployment

The system is fully local, free to run, and designed to mirror real-world RAG service architectures.

🧩 High-level architecture
                ┌──────────────────────┐
                │ conversations.json   │
                │ (ChatGPT export)     │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Offline Ingestion    │
                │  • Chunking          │
                │  • Embeddings        │
                │  • FAISS indexing    │
                └─────────┬────────────┘
                          │
                faiss_index_cache.pkl
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ FastAPI Inference Service        │
        │  • Query endpoint               │
        │  • FAISS retrieval              │
        │  • Prompt construction          │
        └─────────┬────────────────────────┘
                  │
                  ▼
        ┌──────────────────────────────────┐
        │ Local LLM Backend (Ollama)       │
        │  • Mistral Instruct              │
        │  • Runs as separate service      │
        └──────────────────────────────────┘

🧠 Key design decisions (and why they matter)
1️⃣ Offline ingestion vs online inference

Indexing is separated from inference:

Offline (ingest.py)

Parses chat history

Chunks dialogue content

Builds FAISS index

Online (service.py + app.py)

Loads prebuilt index

Serves low-latency queries

This mirrors production RAG systems where indexing is expensive and inference must remain lightweight.

2️⃣ Dialogue-level semantic retrieval

Instead of treating chats as long documents, conversations are chunked at the dialogue level, improving:

contextual relevance

recall for conversational queries

grounding quality in responses

3️⃣ External LLM backend (Ollama)

The LLM runs as a separate service outside the container:

Avoids bloated Docker images

Mirrors microservice separation

Allows swapping local vs API-based LLMs later

The FastAPI service communicates with the LLM via:

http://host.docker.internal:11434

4️⃣ Dockerized, reproducible deployment

The FastAPI service, vector index, and runtime dependencies are packaged into Docker so the system can be launched with:

docker run -p 8000:8000 chatgpt-rag


This ensures reproducibility across machines without re-running ingestion.

📂 Project structure
OLD_CHAT_RAG/
├── app.py                     # FastAPI application
├── service.py                 # Core RAG logic (retrieval + generation)
├── ingest.py                  # Offline ingestion & FAISS indexing
├── conversations.json         # ChatGPT export (user-provided)
├── faiss_index_cache.pkl      # Persisted vector index (generated)
├── Dockerfile                 # Docker deployment
├── requirements.txt           # Python dependencies
├── rag_chatgpt_history.py     # Original interactive prototype
├── helper/                    # Experimental drafts (ignored)
└── README.md

🚀 How to run the project
🔹 Prerequisites

Python 3.10+

Docker Desktop (Windows/macOS/Linux)

Ollama installed and running

A local LLM pulled (e.g. mistral:instruct)

Check Ollama:

ollama list


If needed:

ollama pull mistral

🔹 1. Create a virtual environment (recommended)
python -m venv .env_chat_rag


Activate:

Windows

.env_chat_rag\Scripts\activate


macOS / Linux

source .env_chat_rag/bin/activate


Install dependencies:

pip install -r requirements.txt

🔹 2. Run offline ingestion (one-time)
python ingest.py


This will:

parse the chat export

build embeddings

save faiss_index_cache.pkl

🔹 3. Run locally (without Docker)
uvicorn app:app --reload


Open:

http://localhost:8000/docs


Test /query:

{
  "question": "What ML or AI projects have I worked on?",
  "top_k": 4
}

🔹 4. Run with Docker (recommended)

Build the image:

docker build -t chatgpt-rag .


Run the container:

docker run -p 8000:8000 chatgpt-rag


Then open:

http://localhost:8000/docs


The Dockerized service will:

load the FAISS index

call the external Ollama LLM

return grounded responses

🧪 Example API usage
POST /query

Request

{
  "question": "What ML or AI projects have I worked on?",
  "top_k": 4
}


Response

{
  "question": "What ML or AI projects have I worked on?",
  "answer": "At Orangewood Labs, I worked on projects involving deep learning, reinforcement learning, and transformer-based models..."
}

🛡️ Limitations & scope (by design)

No model training or fine-tuning

No cloud deployment (local-first by intent)

No MLflow (no training loop to track)

No Kubernetes (kept intentionally simple)

These choices keep the project focused, defensible, and interview-safe.

🎯 What this project demonstrates

Practical RAG system design

Vector database lifecycle management

LLM integration and prompt grounding

API-based inference services

Dockerized reproducibility

Debugging across OS, Docker, and networking layers
