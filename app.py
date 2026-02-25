
import os
import tempfile
import streamlit as st

from rag.loader import load_and_chunk
from rag.embedder import embed_and_store, load_vector_store, CHROMA_DIR
from rag.retriever import answer_with_citations


st.set_page_config(
    page_title="Doc Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3147;
    }

    [data-testid="stChatMessage"] {
        background-color: #1a1d27;
        border: 1px solid #2e3147;
        border-radius: 12px;
        margin-bottom: 8px;
        padding: 4px;
    }

    .citation-badge {
        display: inline-block;
        background-color: #2e3147;
        color: #818cf8;
        border: 1px solid #4f46e5;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px;
        font-family: monospace;
    }

    .status-box {
        background-color: #1a1d27;
        border: 1px solid #2e3147;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


with st.sidebar:
    st.markdown("## 📄 Document Q&A")
    st.markdown("*Powered by Llama 3.2 + ChromaDB*")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf","docx"],
        help="Upload any PDF to start asking questions about it."
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:

            with st.spinner(f"Processing **{uploaded_file.name}**..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    st.info("📖 Reading and chunking PDF...")
                    chunks = load_and_chunk(tmp_path)

                    if os.path.exists(CHROMA_DIR):
                        import shutil
                        try:
                            shutil.rmtree(CHROMA_DIR)
                        except Exception as e:
                            st.warning(f"Could not clear old database: {e}")

                    st.info("🧠 Embedding chunks into vector store...")
                    vector_store = embed_and_store(chunks)

                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success(f"✅ Ready! Loaded **{len(chunks)}** chunks.")

                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")

                finally:
                    os.unlink(tmp_path)

        else:
            st.success(f"✅ **{uploaded_file.name}** is loaded.")

    st.divider()

    st.markdown("**Models**")
    st.markdown("""
    <div class="status-box">
        🤖 LLM: <code>llama-3.2-3b</code> via Groq<br>
        🔢 Embeddings: <code>all-MiniLM-L6-v2</code><br>
        🗄️ Vector DB: <code>ChromaDB</code> (local)
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown("## 💬 Ask Your Document")

if not st.session_state.vector_store:
    st.markdown("""
    <div class="status-box" style="text-align:center; padding: 40px;">
        👈 Upload a PDF from the sidebar to get started.<br><br>
        <small>Make sure <code>GROQ_API_KEY</code> is set in your <code>.env</code> file.</small>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                sources_html = "".join(
                    f'<span class="citation-badge">📄 Page {p}</span>'
                    for p in message["sources"]
                )
                st.markdown(
                    f'<div style="margin-top:8px;">Sources: {sources_html}</div>',
                    unsafe_allow_html=True
                )

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_with_citations(
                        question,
                        st.session_state.vector_store
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        sources_html = "".join(
                            f'<span class="citation-badge">📄 Page {p}</span>'
                            for p in sources
                        )
                        st.markdown(
                            f'<div style="margin-top:8px;">Sources: {sources_html}</div>',
                            unsafe_allow_html=True
                        )

                    with st.expander("🔍 View retrieved context", expanded=False):
                        for i, chunk in enumerate(result["chunks"]):
                            st.markdown(f"**Chunk {i+1}**")
                            st.text(chunk[:500])
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(
                        f"❌ Error generating answer: {e}\n\n"
                        "Check that your GROQ_API_KEY is set correctly in your .env file."
                    )
