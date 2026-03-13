from sentence_transformers import CrossEncoder

# Load reranker model once at startup
print("[+] Loading cross-encoder reranker...")
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
from rag.models import reranker_model as reranker
print("[✓] Reranker ready.")


def rerank(query, docs, top_k=4):
    """
    Reorder retrieved documents by semantic relevance.
    """

    if not docs:
        return docs

    # Create query-document pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # Compute relevance scores
    scores = reranker.predict(pairs)

    # Attach scores to documents
    scored_docs = list(zip(docs, scores))

    # Sort by score descending
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Return top documents
    return [doc for doc, score in ranked[:top_k]]