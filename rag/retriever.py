
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from rag.embedder import get_embedding_function, CHROMA_DIR, COLLECTION_NAME

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"   # Fast, free on Groq
TOP_K     = 5


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


def get_llm() -> ChatGroq:

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file or Streamlit secrets."
        )

    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0   # factual, deterministic responses
    )



def get_retriever(vector_store: Chroma):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
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
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
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
