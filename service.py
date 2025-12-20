# service.py
import pickle
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

INDEX_PATH = Path("faiss_index_cache.pkl")

if not INDEX_PATH.exists():
    raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

print("[+] Loading FAISS index...")
with open(INDEX_PATH, "rb") as f:
    vector_store = pickle.load(f)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

print("[+] Loading local LLM (Mistral Instruct)...")
# llm = OllamaLLM(model="mistral:instruct")
llm = OllamaLLM(
    model="mistral:instruct",
    base_url="http://host.docker.internal:11434"
)


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Ayush's memory assistant.\n\n"
        "Relevant past chat excerpts:\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer concisely and factually."
    ),
)

def query_chat_history(question: str, top_k: int = 4) -> str:
    retriever.search_kwargs["k"] = top_k
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    return llm.invoke(formatted_prompt)
