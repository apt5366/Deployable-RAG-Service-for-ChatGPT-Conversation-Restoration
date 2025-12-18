from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3")
response = llm.invoke("Hello Phi-3 from LangChain! How are you today?")
print("💬", response)
