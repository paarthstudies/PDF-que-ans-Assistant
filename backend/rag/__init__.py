"""
RAG package — orchestrates loading, indexing, retrieval, and answer generation.
"""

import os

from dotenv import load_dotenv

# Load API key from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from rag.loader import load_and_chunk
from rag.vectorstore import build_vectorstore, load_vectorstore, index_exists
from rag.retriever import setup_retriever, serialize_retrieved_chunks, extract_sources
from rag.chain import setup_llm, generate_answer

__all__ = [
    "create_chain",
    "load_chain",
    "serialize_retrieved_chunks",
    "extract_sources",
    "generate_answer",
]


def create_chain(file_path):
    """
    Load a document, chunk it, embed it, build a FAISS index, and save it.

    Sets up the retriever and LLM so /chat can retrieve and answer questions.
    """
    chunks = load_and_chunk(file_path)
    vectorstore = build_vectorstore(chunks)
    setup_retriever(vectorstore)
    setup_llm()


def load_chain():
    """
    Load persisted FAISS index on startup.

    Returns True if an index was found and loaded, False otherwise.
    """
    if not index_exists():
        return False

    vectorstore = load_vectorstore()
    setup_retriever(vectorstore)
    setup_llm()
    return True
