# 📄 Document Q&A Assistant

A fully local, privacy-first RAG (Retrieval-Augmented Generation) application that lets you chat with any PDF document. Upload a PDF, ask questions in plain English, and get grounded answers with page citations.

Built with **LangChain**, **Groq**, **HuggingFace**, **ChromaDB**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-red)

🚀 **[Live Demo](https://doc-app-assistant-jvmhvtg7rnkdidptef6ks7.streamlit.app)** ← replace with your URL

---

## 🎯 What It Does

Upload any PDF — a research paper, contract, report, or manual — and ask questions about it in plain English. The app finds the most relevant sections and generates a grounded answer with page citations.

- ✅ **No hallucinations** — the LLM only answers from your document's content
- ✅ **Page citations** — every answer links back to source page numbers
- ✅ **Transparent retrieval** — inspect the exact chunks used to generate each answer
- ✅ **Free to run** — Groq's free tier is fast and generous

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
│  all-MiniLM-L6-v2 converts chunks → vectors │
│  ChromaDB stores vectors in /tmp            │
└─────────────────┬───────────────────────────┘
                  │
     User asks a question
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  retriever.py                               │
│  Question → vector → similarity search      │
│  Top-5 chunks injected into prompt          │
│  Llama 3.3 70B (via Groq) generates answer  │
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
| LLM | Llama 3.3 70B via Groq | Free API, faster than local inference |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Runs on CPU, no API key needed |
| Vector DB | ChromaDB | Lightweight, no server needed |
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
├── .python-version         # Pins Python 3.11 (via uv)
├── requirements.txt
└── .env                    # Not committed — API keys
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- [Groq API key](https://console.groq.com) — free, takes 1 minute to get

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/doc-qa-assistant.git
cd doc-qa-assistant
```

### 2. Set up the Python environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the project root:
```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) → API Keys.

### 4. Run the app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app.py`
4. Under **Advanced Settings → Secrets**, add:
    ```toml
    GROQ_API_KEY = "gsk_your_key_here"
    ```
5. Click **Deploy**

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
chunk_overlap = 100   # overlap between chunks — increase to avoid boundary loss
```

**Retrieval** (`rag/retriever.py`)
```python
TOP_K       = 5      # chunks retrieved per query
temperature = 0      # 0 = factual/deterministic, 1 = creative
LLM_MODEL   = "llama-3.3-70b-versatile"  # swap to llama-3.1-8b-instant for faster responses
```

---

## 🧠 Key Concepts

**RAG (Retrieval-Augmented Generation)** — Instead of fine-tuning a model on your documents, RAG retrieves relevant snippets at query time and injects them into the prompt. The LLM answers using only that context, reducing hallucinations and keeping it grounded in your data.

**Chunking** — LLMs have limited context windows, so documents are split into small overlapping pieces. Overlap ensures sentences that span chunk boundaries aren't lost.

**Embeddings** — Each chunk is converted into a vector that captures its semantic meaning. Similar meanings produce similar vectors, enabling meaning-based search rather than keyword matching.

**Vector Database** — ChromaDB stores chunk vectors and performs fast similarity search to find the most relevant chunks for any query.

---

## 🔮 Possible Improvements

- [ ] Support multiple PDFs simultaneously with document selection
- [ ] Add hybrid search (semantic + keyword BM25) for better retrieval
- [ ] Implement conversation memory for multi-turn follow-up questions
- [ ] Evaluate retrieval quality with the RAGAS framework
- [ ] Add a reranker to improve chunk ranking
- [ ] Export Q&A sessions as a PDF report

---

## 📄 License

MIT License — free to use, modify, and distribute.
