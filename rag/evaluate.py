from rag.retriever import hybrid_search
from rag.reranker import rerank

# test_queries = [
#     ("docker", "docker"),
#     ("telegram bot", "telegram"),
#     ("reinforcement learning", "reinforcement"),
# ]

test_queries = [
    ("docker", "docker"),
    ("telegram bot", "telegram"),
    ("reinforcement learning", "reinforcement"),
    ("fastapi", "fastapi"),
    ("ollama", "ollama"),
    ("docker hub", "docker"),
    ("vector database", "vector"),
    ("rl algorithms", "reinforcement"),
]

TOP_K = 4


def evaluate():

    total = 0
    correct = 0

    for query, keyword in test_queries:

        docs = hybrid_search(query, TOP_K * 2)
        docs = rerank(query, docs, TOP_K)

        for doc in docs:

            total += 1

            if keyword in doc.page_content.lower():
                correct += 1

    precision = correct / total

    print("queries:", len(test_queries))
    print("chunks evaluated:", total)
    print("relevant chunks:", correct)
    print("precision@k:", round(precision, 3))
    # print("precision@k:", precision)
    


if __name__ == "__main__":
    evaluate()