from rag.models import llm


ROUTER_PROMPT = """
You are a router for a chat history assistant.

Your job is to decide which tool should answer the user's question.

Available tools:

retrieve  -> find specific information from past chats
summarize -> summarize conversations about a topic
stats     -> count how many times something appears in chats

Return ONLY one word from this list:
retrieve
summarize
stats

User question:
{question}
"""


def route_query(query: str) -> str:
    """
    Decide which tool should handle the query using the LLM.
    """

    prompt = ROUTER_PROMPT.format(question=query)

    decision = llm.invoke(prompt).strip().lower()

    # safety guard
    if "summarize" in decision:
        return "summarize"

    if "stats" in decision:
        return "stats"

    return "retrieve"