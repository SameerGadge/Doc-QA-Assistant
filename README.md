# 📄 Document Q&A Assistant

A fully local, privacy-first RAG (Retrieval-Augmented Generation) application that lets you chat with any PDF document. No API keys, no data sent to the cloud — everything runs on your machine.

Built with **LangChain**, **Ollama**, **ChromaDB**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Ollama](https://img.shields.io/badge/Ollama-local-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red)

---

## 🎯 What It Does

Upload any PDF — a research paper, contract, report, or manual — and ask questions about it in plain English. The app finds the most relevant sections and generates a grounded answer with page citations.

- ✅ **Fully offline** — no OpenAI or external API needed
- ✅ **Source citations** — every answer links back to page numbers
- ✅ **Transparent retrieval** — inspect the exact chunks used to generate each answer
- ✅ **Persistent vector store** — documents are embedded once and reused across sessions

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│  loader.py                                  │
│  PyMuPDF extracts text page by page         │
│  RecursiveCharacterTextSplitter chunks it   │
└─────────────────┬───────────────────────────┘
                  │  list[Document] chunks
                  ▼
┌─────────────────────────────────────────────┐
│  embedder.py                                │
│  nomic-embed-text converts chunks → vectors │
│  ChromaDB stores vectors on disk            │
└─────────────────┬───────────────────────────┘
                  │
     User asks a question
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  retriever.py                               │
│  Question → vector → similarity search      │
│  Top-5 chunks injected into prompt          │
│  Llama 3.2 generates grounded answer        │
└─────────────────┬───────────────────────────┘
                  │  answer + page citations
                  ▼
           Streamlit UI (app.py)
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| UI | Streamlit | Fast to build, easy to demo |
| LLM | Llama 3.2 (via Ollama) | Free, local, no API key |
| Embeddings | nomic-embed-text (via Ollama) | Best open-source embedder |
| Vector DB | ChromaDB | Local, no server needed |
| PDF Parsing | PyMuPDF | Fast, accurate text extraction |
| RAG Framework | LangChain | Industry standard |

---

## 📁 Project Structure

```
doc-qa-assistant/
│
├── app.py                  # Streamlit UI — chat interface
├── rag/
│   ├── __init__.py
│   ├── loader.py           # PDF loading & chunking
│   ├── embedder.py         # Embeddings + ChromaDB vector store
│   └── retriever.py        # RAG chain — retrieval + answer generation
│
├── chroma_store/           # Auto-created — persisted vector DB
├── requirements.txt
└── .env                    # Not committed — API keys if needed
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- [Ollama](https://ollama.com) — local LLM runtime

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/doc-qa-assistant.git
cd doc-qa-assistant
```

### 2. Install Ollama and pull models

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the required models
ollama pull llama3.2          # LLM (~2GB)
ollama pull nomic-embed-text  # Embedding model (~270MB)
```

### 3. Set up the Python environment

```bash
# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 4. Run the app

```bash
# Terminal 1 — start Ollama server
ollama serve

# Terminal 2 — launch the app
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 💡 How to Use

1. **Upload a PDF** using the sidebar file uploader
2. Wait for the document to be processed (chunked + embedded)
3. **Type a question** in the chat input
4. View the **answer with page citations**
5. Expand **"View retrieved context"** to see exactly which chunks were used

---

## ⚙️ Configuration

Key parameters can be tuned in each module:

**Chunking** (`rag/loader.py`)
```python
chunk_size    = 500   # characters per chunk — increase for denser docs
chunk_overlap = 100   # shared characters between chunks — increase to avoid boundary loss
```

**Retrieval** (`rag/retriever.py`)
```python
TOP_K       = 5      # number of chunks retrieved per query
temperature = 0      # 0 = factual/deterministic, 1 = creative
```

**Models** (`rag/embedder.py`, `rag/retriever.py`)
```python
EMBEDDING_MODEL = "nomic-embed-text"  # swap for mxbai-embed-large for higher quality
LLM_MODEL       = "llama3.2"          # swap for llama3.1:8b for better reasoning
```

---

## 🧠 Key Concepts

**RAG (Retrieval-Augmented Generation)** — Instead of fine-tuning a model on your documents (expensive), RAG retrieves relevant snippets at query time and injects them into the prompt. The LLM answers using only that context, which keeps it grounded and reduces hallucinations.

**Chunking** — LLMs have limited context windows, so documents are split into small overlapping pieces. Overlap ensures sentences that span chunk boundaries aren't lost.

**Embeddings** — Each chunk is converted into a high-dimensional vector that captures its semantic meaning. Similar meanings produce similar vectors, enabling meaning-based search rather than keyword matching.

**Vector Database** — ChromaDB stores chunk vectors on disk and performs fast approximate nearest-neighbour search to find the most relevant chunks for any query.

---

## 🔮 Possible Improvements

- [ ] Support multiple PDFs simultaneously with document selection
- [ ] Add hybrid search (semantic + keyword BM25) for better retrieval
- [ ] Implement conversation memory for multi-turn follow-up questions
- [ ] Evaluate retrieval quality with RAGAS framework
- [ ] Add a reranker (e.g. `cross-encoder/ms-marco-MiniLM`) to improve chunk ranking
- [ ] Export Q&A sessions as a report

---

## 📄 License

MIT License — free to use, modify, and distribute.
