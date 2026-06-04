"""
FAISS vector store — build, save, and load embeddings on disk.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Persisted index lives next to main.py in backend/faiss_store/
FAISS_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_store")


def get_embeddings():
    """Return the embedding model used for FAISS vector search."""
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


def index_exists():
    """Check whether a saved FAISS index is present on disk."""
    return (
        os.path.exists(os.path.join(FAISS_STORE_DIR, "index.faiss"))
        and os.path.exists(os.path.join(FAISS_STORE_DIR, "index.pkl"))
    )


def save_vectorstore(vectorstore):
    """Persist the FAISS index to faiss_store/index.faiss and index.pkl."""
    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_STORE_DIR)


def load_vectorstore():
    """Load a previously saved FAISS index from disk."""
    return FAISS.load_local(
        FAISS_STORE_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_vectorstore(chunks):
    """Create a new FAISS index from document chunks and save it to disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_vectorstore(vectorstore)
    return vectorstore
