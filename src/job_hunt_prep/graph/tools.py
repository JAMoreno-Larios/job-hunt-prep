"""
tools.py

Here we define all tools to be used in our agent

J. A. Moreno
2026
"""

from typing import List, Tuple
from langchain.tools import tool
from langchain_core.documents import Document

from . import llm_setup

# Initialize retriever
retriever = llm_setup.Retriever().retriever

# Define tools
@tool
def search_user_db(query: str, k: int = 5) -> Tuple[str, List[Document]]:
    """
    Semantically searches a database containing user's personal information
    matching the query.

    Args:
        query: Search terms to look for
        k: The maximum amount of desired results
    """
    try:
        retrieved_docs = retriever.invoke(str(query), k=k)
        # Serialize documents for the model
        serialized = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'Unknown')}"
                + f"\n\nContent: {doc.page_content}"
            )
            for doc in retrieved_docs
        )
        # Return serialized and raw documents
        return (serialized, retrieved_docs)

    except Exception:
        Exception("Search turned no elemments")
        return ("No elements found", [])
