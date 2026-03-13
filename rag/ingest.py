# ingest.py
import json
import time
import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -----------------------
# Config
# -----------------------
CHAT_EXPORT = Path("data/conversations.json")
# INDEX_PATH = Path("data/faiss_index_cache.pkl")
INDEX_PATH = Path("data/faiss_index")

if not CHAT_EXPORT.exists():
    raise FileNotFoundError("conversations.json not found")

print("[+] Loading ChatGPT conversations...")
with open(CHAT_EXPORT, "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
conversations = data if isinstance(data, list) else data.get("conversations", [])

for convo in conversations:
    title = convo.get("title", "Untitled Chat")
    for msg in convo.get("mapping", {}).values():
        message = msg.get("message")
        if not message:
            continue
        parts = message.get("content", {}).get("parts", [])
        for part in parts:
            if isinstance(part, str):
                docs.append(Document(page_content=part, metadata={"title": title}))

print(f"[✓] Parsed {len(docs)} chat messages")

# -----------------------
# Chunking
# -----------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
documents = splitter.split_documents(docs)

print(f"[✓] Created {len(documents)} chunks")

# -----------------------
# Embedding + FAISS
# -----------------------
print("[+] Building FAISS index...")
t0 = time.time()

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# vector_store = FAISS.from_documents(documents, embeddings)

# with open(INDEX_PATH, "wb") as f:
#     pickle.dump(vector_store, f)

vector_store = FAISS.from_documents(documents, embeddings)

vector_store.save_local("data/faiss_index")

print(f"[✓] FAISS index saved to {INDEX_PATH} ({time.time() - t0:.1f}s)")
