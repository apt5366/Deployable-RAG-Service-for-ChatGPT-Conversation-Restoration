# service.py
import pickle
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from rag.retriever import hybrid_search
from rag.reranker import rerank
from rag.router import route_query
from rag.tools import count_mentions
import time
from rag.logger import log_request
# INDEX_PATH = Path("data/faiss_index_cache.pkl")

# if not INDEX_PATH.exists():
#     raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

# print("[+] Loading FAISS index...")
# with open(INDEX_PATH, "rb") as f:
#     vector_store = pickle.load(f)

# retriever = vector_store.as_retriever(search_kwargs={"k": 4})

print("[+] Loading local LLM (Mistral Instruct)...")
# llm = OllamaLLM(model="mistral:instruct")
from rag.models import llm as llm
# llm = OllamaLLM(
#     model="mistral:instruct",
#     base_url="http://host.docker.internal:11434"
# )


# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "You are Ayush's memory assistant.\n\n"
#         "Relevant past chat excerpts:\n"
#         "{context}\n\n"
#         "Question: {question}\n"
#         "Answer concisely and factually."
#     ),
# )

# def query_chat_history(question: str, top_k: int = 4) -> str:
#     retriever.search_kwargs["k"] = top_k
#     # docs = retriever.invoke(question)
#     # docs = hybrid_search(question, top_k)
#     docs = hybrid_search(question, top_k * 2)
#     docs = rerank(question, docs, top_k)

#     context = "\n\n".join(d.page_content for d in docs)

#     formatted_prompt = prompt.format(
#         context=context,
#         question=question
#     )

#     return llm.invoke(formatted_prompt)
# 

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Ayush's memory assistant.\n\n"

        "You must ONLY answer using the provided chat excerpts.\n"
        "If the answer is not contained in the context, say:\n"
        "'I could not find that in the chat history.'\n\n"

        "Chat excerpts:\n"
        "{context}\n\n"

        "Question: {question}\n\n"

        "Answer:"
    ),
)

def query_chat_history(question: str, top_k: int = 4) -> str:
    start_time = time.time()
    route = route_query(question)

    # ------------------
    # Retrieval route
    # ------------------
    if route == "retrieve":

        docs = hybrid_search(question, top_k * 2)

        docs = rerank(question, docs, top_k)

        context = "\n\n".join(d.page_content for d in docs)

        formatted_prompt = prompt.format(
            context=context,
            question=question
        )

        answer = llm.invoke(formatted_prompt)

        sources = [doc.page_content[:300] for doc in docs]
        log_request(question, route, start_time)
        return {
            "answer": answer,
            "sources": sources
        }
        # return llm.invoke(formatted_prompt)

    # ------------------
    # Summarization route
    # ------------------
    if route == "summarize":

        docs = hybrid_search(question, top_k * 3)

        context = "\n\n".join(d.page_content for d in docs)

        summary_prompt = f"""
Summarize the following past chats clearly:

{context}

User request: {question}
"""
        log_request(question, route, start_time)
        answer = llm.invoke(summary_prompt)
        sources = [doc.page_content[:300] for doc in docs]

        return {
            "answer": answer,
            "sources": sources
        }
        # return llm.invoke(summary_prompt)

    # ------------------
    # Stats route
    # ------------------
    if route == "stats":

        words = question.lower().split()

        keyword = words[-1]

        count = count_mentions(keyword)
        log_request(question, route, start_time)
        return f"The term '{keyword}' appears in {count} conversations."