from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import create_chain, load_chain, serialize_retrieved_chunks, extract_sources, generate_answer
from rag.retriever import get_retriever
from memory import add_exchange, get_history_text, build_retrieval_query, clear_memory
import os
from dotenv import load_dotenv

# Load API key from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

# On startup, restore FAISS index from disk if it exists
load_chain()


class ChatRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    content: str
    preview: str
    metadata: dict
    page: int | None = None


class SourceCitation(BaseModel):
    file: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    create_chain(file_path)

    # New document = fresh conversation
    clear_memory()

    return {"message": "File uploaded"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if get_retriever() is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a PDF first.",
        )

    # Step 1: Retrieve chunks (use enriched query for follow-up questions)
    retrieval_query = build_retrieval_query(req.question)
    docs = get_retriever().invoke(retrieval_query)

    # Step 2: Serialize chunks and build source citations
    chunks = serialize_retrieved_chunks(docs)
    sources = extract_sources(docs)

    # Step 3: Generate answer with conversation history for context
    history = get_history_text()
    answer = generate_answer(req.question, docs, history)

    # Step 4: Save this exchange for future follow-up questions
    add_exchange(req.question, answer)

    return ChatResponse(answer=answer, sources=sources, retrieved_chunks=chunks)
