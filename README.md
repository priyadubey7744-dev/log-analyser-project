# 📄 RAG Chat with PDF

A **Retrieval-Augmented Generation (RAG)** app that lets you upload any PDF and have a full conversation with it — powered by **Groq LLM (Llama3)** and semantic search.

> Built to demonstrate RAG architecture for AI Agent Development roles.

---

## ✨ Features

- 📎 **Upload any PDF** — research papers, resumes, books, reports
- 🔍 **Semantic Search** — finds the most relevant passages for each question
- 🤖 **Grounded Answers** — LLM only answers from the document (no hallucination)
- 💬 **Full Conversation** — remembers chat history for follow-up questions
- 📎 **Source Transparency** — shows exactly which chunks were used to answer
- ⚡ **Fast** — runs on free Groq API (Llama3-8b)
- 🆓 **No paid embeddings** — uses TF-IDF locally (swap for OpenAI embeddings easily)

---

## 🖼️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF2)
    │
    ▼
Text Chunking (500 chars + 100 overlap)
    │
    ▼
TF-IDF Vectorization → In-Memory Vector Store
    │
    ▼
User Question → Semantic Search → Top 4 Chunks Retrieved
    │
    ▼
[Chunks + Question + History] → Groq LLM → Grounded Answer
```

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/rag-chat-pdf.git
cd rag-chat-pdf
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Get free Groq API key
→ [https://console.groq.com](https://console.groq.com) (free, no credit card)

### 4. Run
```bash
streamlit run app.py
```

---

## 🏗️ Project Structure

```
rag-chat-pdf/
│
├── app.py                  # Streamlit UI — upload, chat, display
├── requirements.txt
├── README.md
│
└── src/
    ├── pdf_processor.py    # PDF text extraction + chunking
    ├── vector_store.py     # TF-IDF embeddings + cosine similarity search
    └── rag_chain.py        # Core RAG pipeline: Retrieve → Augment → Generate
```

---

## 🧠 How RAG Works (Simple Explanation)

**Problem:** LLMs don't know about your specific document.

**RAG Solution:**
1. **Split** the document into small chunks
2. **Store** chunks as vectors (mathematical representations)
3. When you ask a question, **find** the most similar chunks
4. **Inject** those chunks into the LLM prompt as context
5. LLM **answers** based only on that context → no hallucination

**Why this matters:** The LLM doesn't need to be retrained on your data. RAG is faster, cheaper, and keeps answers up-to-date.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web UI |
| **Groq + Llama3** | LLM generation (free API) |
| **PyPDF2** | PDF text extraction |
| **scikit-learn TF-IDF** | Local embeddings (no API cost) |
| **Cosine Similarity** | Semantic chunk retrieval |

---

## 🗺️ Upgrade Path

Want to make this production-grade? Easy swaps:

| Current | Upgrade To |
|---------|-----------|
| TF-IDF | OpenAI `text-embedding-3-small` |
| In-memory store | ChromaDB / Pinecone |
| Single PDF | Multiple PDFs / web URLs |
| Groq Llama3 | GPT-4o / Claude |

---

## 📄 License

MIT — free to use and modify.
