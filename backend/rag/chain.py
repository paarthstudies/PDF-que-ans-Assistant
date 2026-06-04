"""
Answer generation — build prompts and call the LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Set after upload or startup load
_llm = None

PROMPT = ChatPromptTemplate.from_template("""
Answer the question based on the context and conversation history.

Context:
{context}

Conversation History:
{history}

Question:
{question}
""")


def get_llm():
    """Return the active LLM, or None if not initialized yet."""
    return _llm


def setup_llm():
    """Initialize the Gemini LLM used for answer generation."""
    global _llm
    _llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


def format_docs(docs):
    """Join retrieved document chunks into one context string for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(question, docs, history="No previous conversation."):
    """
    Generate an answer using pre-retrieved documents and chat history.

    Retrieval happens separately so we can expose the chunks in the API
    response without calling the retriever twice.
    """
    context = format_docs(docs)
    chain = PROMPT | get_llm()
    response = chain.invoke(
        {"context": context, "history": history, "question": question}
    )
    return response.content[0]["text"]
