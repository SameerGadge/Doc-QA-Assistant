
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


CHROMA_DIR = "/tmp/chroma_store"
COLLECTION_NAME  = "doc_qa"

# Small, fast, high-quality embedding model (~90MB)
# Runs on CPU — no GPU or API key needed
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # CPU is fine for this model
        encode_kwargs={"normalize_embeddings": True}  # cosine similarity works better normalised
    )


def embed_and_store(chunks: list[Document]) -> Chroma:

    print(f"[embedder] Embedding {len(chunks)} chunks via HuggingFace...")

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
            "No existing vector store found and no chunks provided."
        )

    return embed_and_store(chunks)
