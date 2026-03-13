import time


def log_request(query: str, route: str, start_time: float):

    latency = time.time() - start_time

    print(
        f"[QUERY] {query} | route={route} | latency={latency:.2f}s"
    )