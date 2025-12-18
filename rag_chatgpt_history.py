# ============================================
# ChatGPT RAG Memory Vault (LangChain 1.x Compatible, Final)
# Author: Ayush Tiwari
# ============================================

import json
from pathlib import Path
import os
import pickle
import time

# --- STEP 1: Imports ---
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate


# --- STEP 2: Load your ChatGPT export ---
file_path = Path("conversations.json")
if not file_path.exists():
    raise FileNotFoundError("❌ conversations.json not found. Please place it in the same folder as this script.")

print("\n[+] Loading exported ChatGPT conversations...")
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- STEP 3: Parse your messages into documents ---
docs = []
conversations = data if isinstance(data, list) else data.get("conversations", [])
skipped = 0

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
            elif isinstance(part, dict):
                text_repr = json.dumps(part, ensure_ascii=False)
                docs.append(
                    Document(page_content=f"[Non-text content: {text_repr[:100]}...]", metadata={"title": title})
                )
            else:
                skipped += 1

print(f"[✓] Loaded {len(docs)} text entries from {len(conversations)} conversations. Skipped {skipped} non-text parts.")

# --- STEP 4: Split text into chunks ---
print("[+] Splitting text into manageable chunks...")
t0 = time.time()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
documents = splitter.split_documents(docs)
print(f"[✓] Created {len(documents)} text chunks for embedding in {time.time() - t0:.1f}s.\n")

# --- STEP 5: FAISS index setup with caching ---
cache_path = Path("faiss_index_cache.pkl")
print("[+] Preparing FAISS index (with caching)...")

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

if cache_path.exists():
    print("[↻] Loading existing FAISS index from cache...")
    with open(cache_path, "rb") as f:
        vector_store = pickle.load(f)
else:
    print("[+] Creating new FAISS index (this may take a few minutes)...")
    t1 = time.time()
    vector_store = FAISS.from_documents(documents, embeddings)
    with open(cache_path, "wb") as f:
        pickle.dump(vector_store, f)
    print(f"[✓] FAISS index cached for future runs ({time.time() - t1:.1f}s).")

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print("[✓] FAISS index ready!\n")

# --- STEP 6: Initialize your local (not Phi-3 model) rather Mistral Instruct model ---

# print("[+] Initializing local model (Phi-3 via Ollama)...")
# llm = OllamaLLM(model="phi3")
print("[+] Loading Mistral model (7B Instruct)... please wait.")
llm = OllamaLLM(model="mistral:instruct")
print("[✓] Mistral model ready for ChatGPT-style responses.\n")


# --- STEP 7: Create the RAG chain ---
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Ayush's memory assistant. Use the following past chat excerpts to answer:\n\n"
        "{context}\n\nQuestion: {question}\nAnswer concisely and factually."
    ),
)

# Each stage as a RunnableLambda
retrieve_docs = RunnableLambda(lambda x: {
    "context": retriever.invoke(x["question"]),  # fixed API ✅
    "question": x["question"],
})

combine_context = RunnableLambda(lambda x: {
    "context": "\n\n".join([d.page_content for d in x["context"]]),
    "question": x["question"],
})

format_prompt = RunnableLambda(lambda x: {
    "prompt": prompt.format(context=x["context"], question=x["question"])
})

generate_answer = RunnableLambda(lambda x: {
    "answer": llm.invoke(x["prompt"])
})

# Chain together
rag_chain = retrieve_docs | combine_context | format_prompt | generate_answer

print("\n[✓] RAG system is ready! You can now ask questions about your past ChatGPT chats.")
print("Type 'exit' to quit.\n")

# --- STEP 8: Interactive loop ---
while True:
    query = input("Ask about your old chats → ")
    if query.lower() in ["exit", "quit"]:
        print("\nGoodbye, Ayush 👋\n")
        break
    t2 = time.time()
    result = rag_chain.invoke({"question": query})
    print(f"\n💬 {result['answer']}")
    print(f"⏱️  Response time: {time.time() - t2:.1f}s\n")
