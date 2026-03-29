# SimpleRag (OpenAI + Ollama)

This project is a small RAG (Retrieval-Augmented Generation) demo using LangChain:
- It loads a web page, splits/chunks it, builds a FAISS vector index, retrieves relevant chunks, and asks an LLM to answer using only the retrieved context.

## Project Layout

- `open_ai/app.ipynb`: RAG using `OpenAIEmbeddings` + `ChatOpenAI` (`gpt-4o`)
- `ollama/app.ipynb`: RAG using Ollama (Ollama embeddings + `gemma3:4b`)
- `ollama/simple_ollama_rag.py`: Minimal, dependency-light `.py` version of the Ollama RAG (recommended)
- `requirements.txt`: Base Python deps for the OpenAI notebook
- `ollama/requirements.txt`: Python deps for the Ollama notebook/script

## Prerequisites

1. Python 3.10+ (My environment currently uses Python 3.12)
2. A Python environment (venv/conda)
3. Ollama installed (for the Ollama parts)
4. Internet access (the loader fetches a LangSmith documentation page)

## Environment Variables (`.env`)

The OpenAI notebook expects a `.env` file in the project root with:
- `OPENAI_API_KEY`
- `LANGCHAIN_API_KEY` (optional, for LangSmith tracing)
- `LANGCHAIN_PROJECT` (optional)

Do not commit your `.env` file.

## OpenAI Version (Notebook)

From the project root (`d:\SimpleRag`):

1. Install deps:
   - `pip install -r requirements.txt`
2. Ensure `OPENAI_API_KEY` is set in your `.env`
3. Run the notebook:
   - open `open_ai/app.ipynb` in Jupyter/Cursor notebooks and execute cells

The notebook uses:
- `WebBaseLoader` for: `https://docs.smith.langchain.com/tutorials/Administrators/manage_spend`
- `OpenAIEmbeddings`
- `ChatOpenAI(model="gpt-4o")`

## Ollama Version (Minimal `.py` Script)

Recommended: use `ollama/simple_ollama_rag.py`.

### 1) Install Ollama deps
From the project root:
- `pip install -r ollama/requirements.txt`

### 2) Pull required Ollama models
Your script defaults to:
- Embeddings model: `nomic-embed-text`
- LLM model: `gemma3:4b`

Run:
- `ollama pull nomic-embed-text`
- (optional) `ollama pull gemma3:4b` (if not already present)

Verify:
- `ollama list`

### 3) Run the script
- `python ollama\simple_ollama_rag.py`

### Optional Ollama config
Set these env vars before running:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `OLLAMA_LLM_MODEL` (default: `gemma3:4b`)

### If you still see “model not found”
This means the embedding/LLM model name isn’t pulled in Ollama. Pull it with:
- `ollama pull <model_name>`

## Ollama Version (Notebook)

If you want the notebook version (`ollama/app.ipynb`), it should be similar to the OpenAI notebook, but you will still need the Ollama models available in your local Ollama instance.

## Notes / Troubleshooting

- The Ollama `.py` script intentionally avoids `langchain_text_splitters` / `langchain_classic` to reduce dependency issues in some environments.
- LangChain has deprecated `OllamaEmbeddings` in favor of `langchain-ollama`; if you want, we can update the script to use the new package.

