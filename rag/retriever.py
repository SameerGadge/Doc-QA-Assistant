
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from rag.embedder import get_embedding_function, CHROMA_DIR, COLLECTION_NAME

LLM_MODEL = "llama3.2"

TOP_K = 5


RAG_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant that answers questions based strictly on the 
provided document context. 

Rules:
- Only use information from the context below to answer.
- If the context doesn't contain enough information, say: 
  "I couldn't find relevant information in the document for this question."
- Be concise and direct. Avoid unnecessary filler phrases.
- Do not make up or infer information beyond what's in the context.

Context:
{context}

Question:
{question}

Answer:
""")


def get_llm() -> OllamaLLM:
    return OllamaLLM(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        temperature=0  # 0 = focused/factual, 1 = creative/varied
    )


def get_retriever(vector_store: Chroma):

    return vector_store.as_retriever(
        search_type="similarity",   # cosine similarity search
        search_kwargs={"k": TOP_K}  # return top 5 chunks
    )


def format_docs(docs) -> str:

    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vector_store: Chroma):

    retriever = get_retriever(vector_store)
    llm = get_llm()

    rag_chain = (
        {
            # Left side: retrieve chunks and format them into context string
            "context": retriever | format_docs,
            # Right side: pass the question through unchanged
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT       # inject context + question into the prompt template
        | llm              # send filled prompt to Llama 3.2 via Ollama
        | StrOutputParser() # parse LLM output to a plain string
    )

    return rag_chain


def answer_with_citations(question: str, vector_store: Chroma) -> dict:

    chain = build_rag_chain(vector_store)
    answer = chain.invoke(question)

    retriever = get_retriever(vector_store)
    source_docs = retriever.invoke(question)

    seen = set()
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        if page not in seen:
            seen.add(page)
            sources.append(page)

    chunks = [doc.page_content for doc in source_docs]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "chunks": chunks
    }


if __name__ == "__main__":
    from embedder import load_vector_store

    print("Loading vector store...")
    store = load_vector_store()

    print("\nReady! Type your questions (or 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\n[Thinking...]\n")
        result = answer_with_citations(question, store)

        print(f"Answer:\n{result['answer']}")
        print(f"\nSources: Pages {result['sources']}")
        print("-" * 60)
