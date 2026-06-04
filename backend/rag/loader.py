"""
Document loading and chunking.

Loads PDF, TXT, or DOCX files and splits them into smaller chunks
for embedding and retrieval.
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chunking settings — adjust these to change how documents are split
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_file(file_path):
    """Load a PDF, TXT, or DOCX file and return LangChain Document objects."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError("Unsupported file format")

    return loader.load()


def chunk_documents(docs):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def load_and_chunk(file_path):
    """Load a file from disk and return chunked documents."""
    docs = load_file(file_path)
    return chunk_documents(docs)
