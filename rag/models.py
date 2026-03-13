from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

print("[+] Loading LLM...")
# llm = OllamaLLM(model="mistral:instruct")
llm = OllamaLLM(
    model="mistral:instruct",
    base_url="http://host.docker.internal:11434"
)

print("[+] Loading reranker...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("[✓] Models ready.")