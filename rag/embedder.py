import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

COLLECTION_NAME = "doc_qa"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def embed_and_store(chunks: list[Document]) -> Chroma:
    print(f"[embedder] Embedding {len(chunks)} chunks via HuggingFace...")

    embedding_fn = get_embedding_function()

    # ✅ EphemeralClient = in-memory, no disk writes, works everywhere
    client = chromadb.EphemeralClient()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        client=client          # ← pass client directly, drop persist_directory
    )

    print(f"[embedder] Stored {len(chunks)} vectors in memory.")
    return vector_store


def get_or_create_vector_store(chunks: list[Document] = None) -> Chroma:
    if chunks is None:
        raise ValueError("No chunks provided to build vector store.")
    return embed_and_store(chunks)