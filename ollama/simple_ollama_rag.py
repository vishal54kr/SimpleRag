import os
import urllib.request

from bs4 import BeautifulSoup

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


URL = "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend"
QUERY = "LangSmith has two usage limits: total traces and extended"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:4b")


def fetch_page_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements when possible
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Basic text extraction. Separator helps keep chunk boundaries readable.
    return soup.get_text(separator="\n", strip=True)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Simple deterministic chunker to avoid extra dependencies.
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be > chunk_overlap")

    step = chunk_size - chunk_overlap
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


def main():
    print("Fetching document...")
    page_text = fetch_page_text(URL)

    print("Chunking text...")
    chunks = chunk_text(page_text, chunk_size=1000, chunk_overlap=200)
    documents = [Document(page_content=c) for c in chunks]

    print("Creating embeddings + FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = FAISS.from_documents(documents, embeddings)

    print("Retrieving relevant context...")
    retrieved_docs = vectorstore.similarity_search(QUERY, k=4)
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    prompt = f"""Answer the following question based only on the provided context.
<context>
{context}
</context>

Question: {QUERY}
"""

    print("Calling Ollama LLM...")
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    resp = llm.invoke([HumanMessage(content=prompt)])

    print("\n--- Answer ---")
    print(getattr(resp, "content", str(resp)))


if __name__ == "__main__":
    main()

