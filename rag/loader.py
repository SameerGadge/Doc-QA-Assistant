import fitz  #PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf(file_path: str) -> list[Document]:

    documents = []

    with fitz.open(file_path) as pdf:
        for page_number in range(len(pdf)):
            page = pdf[page_number]

            text = page.get_text()

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_number + 1  # human-readable (1-indexed)
                    }
                )
            )

    print(f"[loader] Loaded {len(documents)} pages from '{file_path}'")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"[loader] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return chunks


def load_and_chunk(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[Document]:

    pages = load_pdf(file_path)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    return chunks

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag/loader.py <path_to_pdf_docx>")
        sys.exit(1)

    test_path = sys.argv[1]
    chunks = load_and_chunk(test_path)

    print("\n--- Sample Chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}] Page {chunk.metadata['page']}")
        print(chunk.page_content[:300])
        print("...")
