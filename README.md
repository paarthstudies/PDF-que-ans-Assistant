# PDF Question Answering Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application built with FastAPI, LangChain, Google Gemini, and FAISS.

Upload a document, ask questions in natural language, and get answers grounded in your file — with retrieved chunks, source citations, and conversational follow-ups visible in every response.

This project is designed as a **learning and interview-ready** codebase. The goal is clarity over complexity, not production-scale infrastructure.

---

## Features (Phase 1)

| Feature | What it does |
|---------|--------------|
| **PDF Upload** | Load PDF, TXT, or DOCX files and build a searchable vector index |
| **Retrieval Debug Viewer** | See exactly which chunks were retrieved for each question |
| **Top-K Retrieval** | Control how many chunks are retrieved (`TOP_K = 3`) |
| **Source Citations** | Every answer includes filename and page number |
| **Persistent FAISS Storage** | Vector index survives server restarts |
| **Conversational Memory** | Follow-up questions like "Explain it simply" work in context |

---

## How RAG Works (Simple Explanation)

Large language models do not know your private documents. RAG solves this by:

1. **Loading** your document and splitting it into chunks
2. **Embedding** each chunk into a numerical vector
3. **Storing** vectors in FAISS for fast similarity search
4. **Retrieving** the most relevant chunks when you ask a question
5. **Generating** an answer using only those chunks as context

```
PDF Upload → Document Loader → Text Splitter → Embeddings → FAISS
                                                              ↓
User Question → Retriever → Top-K Chunks → Prompt + Gemini → Answer
```

---

## Project Structure

```
Rag phase 1/
├── .env                    # GEMINI_API_KEY (not committed)
├── README.md
└── backend/
    ├── main.py             # FastAPI routes (/upload, /chat)
    ├── memory.py           # Simple session chat history
    ├── requirements.txt
    ├── uploads/            # Uploaded files
    ├── faiss_store/        # Persisted index (created after first upload)
    │   ├── index.faiss
    │   └── index.pkl
    └── rag/                # RAG pipeline (one file per responsibility)
        ├── __init__.py     # Orchestrates upload + startup load
        ├── loader.py       # Document loading and chunking
        ├── vectorstore.py  # FAISS build, save, and load
        ├── retriever.py    # Top-K retrieval, citations, debug chunks
        └── chain.py        # Prompt construction and answer generation
```

---

## Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

---

## Setup

### 1. Clone or open the project

```bash
cd "c:\projects\Rag phase 1"
```

### 2. Create your `.env` file

In the **project root** (same level as `backend/`), create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

The backend loads this automatically. **Never commit your API key.**

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the server

```bash
python -m uvicorn main:app --reload
```

The API runs at **http://127.0.0.1:8000**

Interactive docs: **http://127.0.0.1:8000/docs**

---

## API Reference

### POST `/upload`

Upload a document and build (or rebuild) the vector index.

**Request:** multipart form with a `file` field (PDF, TXT, or DOCX)

**Response:**
```json
{
  "message": "File uploaded"
}
```

Uploading a new file clears conversation memory and overwrites the FAISS index.

---

### POST `/chat`

Ask a question about the uploaded document.

**Request:**
```json
{
  "question": "What is deadlock?"
}
```

**Response:**
```json
{
  "answer": "Deadlock is a condition where two or more processes...",
  "sources": [
    {
      "file": "OperatingSystems.pdf",
      "page": 12
    }
  ],
  "retrieved_chunks": [
    {
      "content": "Deadlock occurs when two or more threads...",
      "preview": "Deadlock occurs when two or more threads...",
      "metadata": {
        "source": "uploads/OperatingSystems.pdf",
        "page": 11
      },
      "page": 12
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `answer` | LLM-generated answer based on retrieved context |
| `sources` | Deduplicated list of file + page citations |
| `retrieved_chunks` | Full debug view of what retrieval selected |
| `metadata.page` | 0-based page index from LangChain |
| `page` (top-level) | 1-based page number for human display |

---

## Usage Examples

### Using Swagger UI

1. Open http://127.0.0.1:8000/docs
2. Call **POST /upload** and select a PDF
3. Call **POST /chat** with your question
4. Inspect `retrieved_chunks` and `sources` in the response

### Using curl

```bash
# Upload
curl -X POST http://127.0.0.1:8000/upload -F "file=@your-document.pdf"

# Ask a question
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is deadlock?\"}"

# Follow-up (uses conversational memory)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Explain it simply.\"}"
```

### Test persistence (FAISS)

1. Upload a document and ask a question — confirm it works
2. Stop the server (`Ctrl+C`)
3. Restart: `python -m uvicorn main:app --reload`
4. Ask a question **without re-uploading** — retrieval should still work

The index is loaded from `backend/faiss_store/` on startup.

---

## Configuration

All key settings live in the `rag/` package:

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| `TOP_K` | `rag/retriever.py` | `3` | Number of chunks retrieved per question |
| `CHUNK_SIZE` | `rag/loader.py` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `rag/loader.py` | `200` | Overlap between consecutive chunks |
| Embedding model | `rag/vectorstore.py` | `gemini-embedding-2` | Google embedding model |
| LLM | `rag/chain.py` | `gemini-3-flash-preview` | Answer generation model |

To change retrieval count, edit `TOP_K` in `rag/retriever.py` and re-upload your document:

```python
TOP_K = 3  # change to 5, re-upload, and retrieval returns 5 chunks
```

---

## Architecture Walkthrough

### Document processing (`create_chain` in `rag/__init__.py`)

```
loader.load_and_chunk() → vectorstore.build_vectorstore() → retriever.setup_retriever()
```

### Question answering (`/chat` in `main.py`)

```
build_retrieval_query() → retriever.invoke() → serialize_retrieved_chunks()
                                              → extract_sources()
                                              → generate_answer() with get_history_text()
                                              → add_exchange()
```

### Conversational memory (`memory.py`)

- Stores Q&A pairs in a simple Python list
- **Prompt memory:** history is injected into the LLM prompt
- **Retrieval memory:** follow-ups prepend the last question so FAISS still finds relevant chunks
- Cleared automatically on new upload

Example:
```
Q1: What is deadlock?
Q2: Explain it simply.
```
For Q2, retrieval searches: `"What is deadlock? Explain it simply."`

---

## Interview Talking Points

After building this project, you should be able to explain:

### 1. Document Loading
PyPDFLoader extracts text page-by-page and attaches metadata (`source`, `page`) to each document.

### 2. Chunking
Large documents are split into ~1000-character chunks with 200-character overlap so context is not lost at boundaries.

### 3. Embeddings
Each chunk is converted to a dense vector. Semantically similar text produces similar vectors.

### 4. FAISS
Facebook AI Similarity Search stores vectors and finds the nearest neighbors to a query vector in milliseconds.

### 5. Retrieval
The user's question is embedded and compared against stored chunk vectors. Top-K most similar chunks are returned.

### 6. Source Citations
Metadata from retrieved chunks (filename, page) is extracted and returned so users can verify answers.

### 7. Memory
Previous questions and answers are stored in session memory. Follow-up questions use both prompt history and enriched retrieval queries.

### 8. Prompt Construction
Retrieved chunks are joined into a context string. The LLM is instructed to answer only from that context plus conversation history.

### 9. Answer Generation
Gemini receives the prompt and produces a natural language answer grounded in retrieved evidence.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No document uploaded yet` | Upload a file via `/upload` first, or ensure `faiss_store/` exists from a prior upload |
| API key errors | Check `.env` in project root contains `GEMINI_API_KEY=...` |
| Empty or wrong answers | Inspect `retrieved_chunks` — if retrieval missed the topic, try rephrasing the question |
| Follow-up fails | Memory is session-only; restarting the server clears history |
| Index not persisting | Confirm `backend/faiss_store/index.faiss` and `index.pkl` exist after upload |

---

## What Is NOT in Phase 1

These are intentionally deferred for later learning phases:

- Hybrid retrieval (BM25 + dense)
- Re-ranking
- LangGraph / multi-agent systems
- Redis, Kafka, Celery, Kubernetes
- Multi-user authentication
- Production deployment

---

## Tech Stack

- **FastAPI** — REST API
- **LangChain** — document loading, chunking, retrieval pipeline
- **Google Gemini** — embeddings and answer generation
- **FAISS** — local vector database
- **python-dotenv** — environment variable management

---

## License

Learning project — use freely for study and portfolio purposes.
