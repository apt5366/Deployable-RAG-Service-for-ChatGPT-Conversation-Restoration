from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import re

# INDEX_PATH = Path("data/faiss_index_cache.pkl")

# -------------------------
# Load FAISS vector store
# -------------------------

# with open(INDEX_PATH, "rb") as f:
#     vector_store = pickle.load(f)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

INDEX_PATH = "data/faiss_index"

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


# Extract documents from FAISS docstore
documents = vector_store.docstore._dict
doc_list = list(documents.values())

# -------------------------
# Better tokenizer
# -------------------------
def tokenize(text: str):
    """
    Tokenize text into lowercase words using regex.
    Handles punctuation and code-like tokens better than split().
    """
    return re.findall(r"\w+", text.lower())


# Prepare BM25 corpus
texts = [doc.page_content for doc in doc_list]
tokenized_corpus = [tokenize(text) for text in texts]

bm25 = BM25Okapi(tokenized_corpus)

# FAISS retriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# -------------------------
# Hybrid Search
# -------------------------
def hybrid_search(query: str, k: int = 4):

    # --------
    # FAISS search
    # --------
    vector_docs = vector_retriever.invoke(query)

    # --------
    # BM25 search
    # --------
    tokenized_query = tokenize(query)

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_docs = [doc_list[i] for i in bm25_top_indices]

    # --------
    # Merge results
    # --------
    combined = vector_docs + bm25_docs

    # Remove duplicates
    unique_docs = {doc.page_content: doc for doc in combined}

    return list(unique_docs.values())[:k]