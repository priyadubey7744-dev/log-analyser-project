# 🤖 RAG Chat with PDF — AI Agent Development Assignment

**Name:** Your Name  
**Role Applied:** AI Agent Development Intern  

---

## 📎 Assignment Answers

### Q1. GitHub Link of an AI/ML Project
This repository IS the project!  
🔗 https://github.com/priyadubey7744-dev/ragchatpdf

**What it does:**
- Upload any PDF document
- Ask questions about it
- AI answers using only that document (no hallucination)
- Built using RAG architecture + Groq LLM (Llama3)

---

### Q2. How to Evaluate if Candidate Answer is Correct in an AI Mock Interview Agent

I would use 3 methods combined:

**1. LLM-as-Judge:**
Pass the question + candidate answer + ideal answer to an LLM
with a scoring rubric. Ask it to rate 1–10 on correctness,
depth, and clarity.

**2. Semantic Similarity:**
Convert candidate answer and ideal answer into embeddings,
then measure cosine similarity. High similarity = good answer.

**3. Key Concept Check:**
For technical questions, verify that must-have concepts
are mentioned in the answer.

Example code:
```python
def evaluate_answer(question, candidate_answer, role):
    prompt = f"""
    You are interviewing for {role}.
    Question: {question}
    Answer: {candidate_answer}
    Score 1-10 and explain why. Return JSON.
    """
    return llm.invoke(prompt)
```

---

### Q3. What is RAG and How to Use it in an Interview Agent

**RAG = Retrieval-Augmented Generation**

Simple explanation:
Imagine an LLM is a smart student who read many books
but doesn't know YOUR specific content.
RAG gives the LLM a "cheat sheet" of relevant information
at the time of answering — so it gives accurate answers
instead of guessing.

**How RAG works:**
1. Store documents as vectors in a database
2. User asks a question
3. System finds the most similar chunks (retrieval)
4. Chunks are injected into the LLM prompt (augmentation)
5. LLM generates a grounded answer (generation)

**In an Interview Agent, RAG can:**
- Store ideal answers → retrieve closest one to evaluate responses
- Pull role-specific questions from a question bank
- Give personalized feedback by comparing weak answers
  to strong reference answers

**This project demonstrates RAG:**
- PDF is chunked and stored as TF-IDF vectors
- User question → semantic search → top 4 chunks retrieved
- Chunks injected into Groq LLM → grounded answer generated

---

## 🚀 How to Run This Project
```bash
pip install -r requirements.txt
streamlit run app.py
```
Get free API key at: https://console.groq.com

---

## 🏗️ Project Structure
```
rag-chat-pdf/
├── app.py                 # Streamlit UI
├── requirements.txt
└── src/
    ├── pdf_processor.py   # PDF extraction + chunking
    ├── vector_store.py    # TF-IDF embeddings + search
    └── rag_chain.py       # RAG pipelinewith a scoring rubric. Ask it to rate 1–10 on correctness,
depth, and clarity.

**2. Semantic Similarity:**
Convert candidate answer and ideal answer into embeddings,
then measure cosine similarity. High similarity = good answer.

**3. Key Concept Check:**
For technical questions, verify that must-have concepts
are mentioned in the answer.

Example code:
```python
def evaluate_answer(question, candidate_answer, role):
    prompt = f"""
    You are interviewing for {role}.
    Question: {question}
    Answer: {candidate_answer}
    Score 1-10 and explain why. Return JSON.
    """
    return llm.invoke(prompt)
```

---

### Q3. What is RAG and How to Use it in an Interview Agent

**RAG = Retrieval-Augmented Generation**

Simple explanation:
Imagine an LLM is a smart student who read many books
but doesn't know YOUR specific content.
RAG gives the LLM a "cheat sheet" of relevant information
at the time of answering — so it gives accurate answers
instead of guessing.

**How RAG works:**
1. Store documents as vectors in a database
2. User asks a question
3. System finds the most similar chunks (retrieval)
4. Chunks are injected into the LLM prompt (augmentation)
5. LLM generates a grounded answer (generation)

**In an Interview Agent, RAG can:**
- Store ideal answers → retrieve closest one to evaluate responses
- Pull role-specific questions from a question bank
- Give personalized feedback by comparing weak answers
  to strong reference answers

**This project demonstrates RAG:**
- PDF is chunked and stored as TF-IDF vectors
- User question → semantic search → top 4 chunks retrieved
- Chunks injected into Groq LLM → grounded answer generated

---

## 🚀 How to Run This Project
```bash
pip install -r requirements.txt
streamlit run app.py
```
Get free API key at: https://console.groq.com

---

## 🏗️ Project Structure
```
rag-chat-pdf/
├── app.py                 # Streamlit UI
├── requirements.txt
└── src/
    ├── pdf_processor.py   # PDF extraction + chunking
    ├── vector_store.py    # TF-IDF embeddings + search
    └── rag_chain.py       # RAG pipeline
```
```

---

5. Scroll down → tap **"Commit changes"** ✅

---

### 🎯 Now Submit Just ONE Link:
```
https://github.com/YOUR_USERNAME/rag-chat-pdf
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
