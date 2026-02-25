
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_store"

COLLECTION_NAME = "doc_qa"

EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding_function() -> OllamaEmbeddings:

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"  # Default Ollama server address
    )


def embed_and_store(chunks: list[Document]) -> Chroma:

    print(f"[embedder] Embedding {len(chunks)} chunks via Ollama...")
    print(f"[embedder] Make sure 'ollama serve' is running!")

    embedding_fn = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(f"[embedder] Stored {len(chunks)} vectors in '{CHROMA_DIR}'")
    return vector_store


def load_vector_store() -> Chroma:

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No vector store found at '{CHROMA_DIR}'. "
            "Please run embed_and_store() first."
        )

    embedding_fn = get_embedding_function()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR
    )

    print(f"[embedder] Loaded existing vector store from '{CHROMA_DIR}'")
    return vector_store


def get_or_create_vector_store(chunks: list[Document] = None) -> Chroma:

    if os.path.exists(CHROMA_DIR):
        return load_vector_store()

    if chunks is None:
        raise ValueError(
            "No existing vector store found and no chunks provided. "
            "Pass chunks to create a new store."
        )

    return embed_and_store(chunks)


if __name__ == "__main__":
    import sys
    from rag.loader import load_and_chunk

    if len(sys.argv) < 2:
        print("Usage: uv run python rag/embedder.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Load and chunk the PDF
    chunks = load_and_chunk(pdf_path)

    # Embed and store
    store = embed_and_store(chunks)

    # Run a quick similarity search to verify it works
    test_query = "What is this document about?"
    results = store.similarity_search(test_query, k=3)

    print(f"\n--- Top 3 results for: '{test_query}' ---")
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}] Page {doc.metadata.get('page', '?')}")
        print(doc.page_content[:300])
        print("...")