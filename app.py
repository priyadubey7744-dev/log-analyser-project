import streamlit as st
from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .chat-bubble-user {
        background: #2d3250;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #eaeaea;
        text-align: right;
    }
    .chat-bubble-ai {
        background: #1a1f35;
        border-radius: 12px 12px 12px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #eaeaea;
        border-left: 3px solid #7c83fd;
    }
    .source-box {
        background: #0f1117;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #aaa;
        border: 1px solid #2d3250;
        margin-top: 6px;
    }
    .stButton > button {
        background: #7c83fd;
        color: white;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("📄 RAG Chat with PDF")
st.markdown("Upload any PDF and chat with it using AI. Answers are grounded in your document.")

col_sidebar, col_chat = st.columns([1, 2])

# ── SIDEBAR: Upload + Settings ────────────────────────────────────────────────
with col_sidebar:
    st.subheader("⚙️ Setup")

    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="Get free key at console.groq.com"
    )

    uploaded_file = st.file_uploader("📎 Upload PDF", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    # Extract text
                    raw_text = extract_text_from_pdf(uploaded_file)

                    # Chunk it
                    chunks = chunk_text(raw_text)

                    # Build vector store
                    vs = VectorStore()
                    vs.build(chunks)

                    st.session_state.vector_store = vs
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {len(chunks)} chunks from **{uploaded_file.name}**")

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    if st.session_state.pdf_name:
        st.info(f"📘 Active: **{st.session_state.pdf_name}**")

    st.divider()
    st.markdown("### 💡 How RAG Works")
    st.markdown("""
1. **PDF → Text** extraction  
2. Text split into **chunks**  
3. Chunks stored as **embeddings** in vector DB  
4. Your question → **semantic search** → top chunks retrieved  
5. Retrieved chunks + question → **LLM generates answer**
""")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── CHAT PANEL ────────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Chat")

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source chunks used"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f'<div class="source-box">**Chunk {i}:** {src[:300]}...</div>', unsafe_allow_html=True)

    # Input
    if not st.session_state.vector_store:
        st.warning("⬅️ Please upload a PDF and enter your API key first.")
    else:
        question = st.text_input(
            "Ask a question about your PDF...",
            placeholder="e.g. What is the main topic of this document?",
            key="question_input"
        )

        if st.button("Send ➤", use_container_width=True) and question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Searching document and generating answer..."):
                rag = RAGChain(api_key=api_key, vector_store=st.session_state.vector_store)
                answer, sources = rag.answer(question, st.session_state.chat_history)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            st.rerun()
