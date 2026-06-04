"""
Retrieval — find relevant chunks and format them for the API response.
"""

import os

# How many chunks to retrieve per question.
# Change this one value to control retrieval behavior.
TOP_K = 3

# Set after upload or startup load; None until a document is indexed.
_retriever = None


def get_retriever():
    """Return the active retriever, or None if no document is indexed yet."""
    return _retriever


def setup_retriever(vectorstore):
    """Create a retriever from the FAISS vector store."""
    global _retriever
    _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def serialize_retrieved_chunks(docs):
    """
    Turn LangChain Documents into debug-friendly JSON for the API response.

    Each chunk includes full text, a short preview, raw metadata, and a
    human-readable page number (1-based) when available.
    """
    chunks = []
    for doc in docs:
        content = doc.page_content
        # PyPDFLoader stores page as 0-based int in metadata["page"]
        page = doc.metadata.get("page")
        chunks.append(
            {
                "content": content,
                "preview": content[:200] + ("..." if len(content) > 200 else ""),
                "metadata": doc.metadata,
                "page": page + 1 if page is not None else None,
            }
        )
    return chunks


def extract_sources(docs):
    """
    Build source citations from retrieved documents.

    Each citation includes the filename and page number (1-based for PDFs).
    Duplicate file+page pairs are skipped so the list stays readable.
    """
    sources = []
    seen = set()

    for doc in docs:
        source_path = doc.metadata.get("source", "")
        filename = os.path.basename(source_path) if source_path else "unknown"

        # PyPDFLoader stores page as 0-based; convert to 1-based for citations
        page_raw = doc.metadata.get("page")
        page = page_raw + 1 if page_raw is not None else None

        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)

        sources.append({"file": filename, "page": page})

    return sources
