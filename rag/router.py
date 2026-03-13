def route_query(query: str) -> str:
    """
    Decide which tool should handle the query.
    """

    q = query.lower()

    if "summarize" in q or "summary" in q:
        return "summarize"

    if "how many" in q or "count" in q or "number of" in q:
        return "stats"

    return "retrieve"